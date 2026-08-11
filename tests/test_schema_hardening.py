from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from evolsa.core import (
    Budget,
    EffectScope,
    OperatorLifecycle,
    OperatorPassport,
    OperatorVerification,
)
from evolsa.schemas import (
    load_operator_passport_schema,
    operator_passport_document,
    operator_passport_from_document,
    operator_passport_schema_bytes,
)


_EVIDENCE_ID = "a" * 64
_GENERATOR_DIGEST = "b" * 64


def _passport_values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "name": "repair-route",
        "version": "1.2.3-alpha.1+build.7",
        "intent": "repair a bounded routing artifact",
        "inputs": ("task", "artifact"),
        "preconditions": ("artifact_exists",),
        "expected_effects": (EffectScope.TASK_ARTIFACT,),
        "forbidden_effects": (EffectScope.EVALUATOR, EffectScope.PROMOTION_GATE),
        "context_policy": "branch-local-v1",
        "generator_type": "prompt_template",
        "generator_ref": f"sha256:{_GENERATOR_DIGEST}",
        "budget": Budget(
            max_tokens=100,
            max_exec_seconds=4.0,
            max_attempts=2,
            max_cost=0.5,
        ),
        "verification": OperatorVerification(
            baseline="improve@1.0.0",
            primary_metric="paired_lift",
            guardrails=("valid_rate", "cost"),
            minimum_tasks=4,
        ),
        "parents": ("improve@1.0.0",),
        "known_failures": ("unsupported_media",),
        "lifecycle": OperatorLifecycle.CANDIDATE,
        "evidence_ids": (_EVIDENCE_ID,),
        "signature": "",
    }
    values.update(overrides)
    return values


def _document() -> dict[str, Any]:
    return operator_passport_document(OperatorPassport(**_passport_values()))


def _set_path(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


class OperatorPassportSchemaHardeningTests(unittest.TestCase):
    def test_repository_and_packaged_schemas_are_identical_and_neutral(self) -> None:
        project = Path(__file__).resolve().parents[1]
        repository_schema = (project / "schemas" / "operator-passport.schema.json").read_bytes()
        self.assertEqual(repository_schema, operator_passport_schema_bytes())
        self.assertEqual(
            load_operator_passport_schema()["$id"],
            "urn:evolsa:schema:operator-passport:v1",
        )

    def test_strict_document_round_trip(self) -> None:
        document = _document()
        parsed = operator_passport_from_document(document)
        self.assertEqual(operator_passport_document(parsed), document)
        self.assertEqual(parsed.generator_ref, f"sha256:{_GENERATOR_DIGEST}")

    def test_parser_does_not_coerce_json_scalar_types(self) -> None:
        invalid_scalars = (
            (("name",), 7),
            (("version",), 1),
            (("intent",), True),
            (("context_policy",), 1),
            (("generator", "type"), 1),
            (("generator", "ref"), False),
            (("budget", "max_tokens"), True),
            (("budget", "max_tokens"), 2.0),
            (("budget", "max_exec_seconds"), False),
            (("budget", "max_exec_seconds"), "4"),
            (("budget", "max_attempts"), True),
            (("budget", "max_cost"), "0.5"),
            (("verification", "baseline"), 1),
            (("verification", "primary_metric"), False),
            (("verification", "minimum_tasks"), True),
            (("signature",), 12),
            (("lifecycle",), 1),
        )
        for path, value in invalid_scalars:
            with self.subTest(path=path, value=value):
                document = _document()
                _set_path(document, path, value)
                with self.assertRaises((TypeError, ValueError)):
                    operator_passport_from_document(document)

        with self.assertRaises(TypeError):
            operator_passport_from_document(tuple(_document().items()))  # type: ignore[arg-type]

    def test_parser_rejects_invalid_identifiers_enums_and_unsafe_refs(self) -> None:
        invalid_values = (
            (("name",), "RepairRoute"),
            (("name",), "x"),
            (("version",), "01.0.0"),
            (("version",), "1.0.0-01"),
            (("version",), "1.0"),
            (("intent",), "short"),
            (("generator", "type"), "shell"),
            (("lifecycle",), "approved"),
            (("expected_effects",), ["evaluator"]),
            (("forbidden_effects",), ["network"]),
            (("evidence_ids",), ["A" * 64]),
            (("evidence_ids",), ["a" * 63]),
        )
        for path, value in invalid_values:
            with self.subTest(path=path, value=value):
                document = _document()
                _set_path(document, path, value)
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)

        unsafe_refs = (
            "/absolute/operator.txt",
            "../operator.txt",
            "operators/../operator.txt",
            r"C:\\operators\\operator.txt",
            r"\\server\\operator.txt",
            "https://example.com/operator.txt",
            "file:operator.txt",
            "sha256:" + "A" * 64,
            "sha256:short",
        )
        for reference in unsafe_refs:
            with self.subTest(reference=reference):
                document = _document()
                document["generator"]["ref"] = reference
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)

    def test_parser_rejects_non_unique_or_non_string_arrays(self) -> None:
        invalid_arrays = (
            ("inputs", ["task", "task"]),
            ("inputs", ["task", 1]),
            ("preconditions", ["ready", "ready"]),
            ("expected_effects", ["task_artifact", "task_artifact"]),
            ("forbidden_effects", ["evaluator", "evaluator"]),
            ("parents", ["parent@1.0.0", "parent@1.0.0"]),
            ("known_failures", ["failure", "failure"]),
            ("evidence_ids", [_EVIDENCE_ID, _EVIDENCE_ID]),
        )
        for field, value in invalid_arrays:
            with self.subTest(field=field):
                document = _document()
                document[field] = value
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)

        document = _document()
        document["verification"]["guardrails"] = ["cost", "cost"]
        with self.assertRaises(ValueError):
            operator_passport_from_document(document)
        document = _document()
        document["inputs"] = ()
        with self.assertRaises(ValueError):
            operator_passport_from_document(document)

    def test_parser_requires_exact_nested_shapes_and_ranges(self) -> None:
        document = _document()
        document[1] = "not-a-json-object-key"  # type: ignore[index]
        with self.assertRaises(ValueError):
            operator_passport_from_document(document)
        document = _document()
        document["budget"][1] = 1  # type: ignore[index]
        with self.assertRaises(ValueError):
            operator_passport_from_document(document)

        for nested in ("budget", "verification", "generator"):
            with self.subTest(nested=nested, change="extra"):
                document = _document()
                document[nested]["unexpected"] = 1
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)
            with self.subTest(nested=nested, change="missing"):
                document = _document()
                document[nested].pop(next(iter(document[nested])))
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)

        invalid_ranges = (
            (("budget", "max_tokens"), -1),
            (("budget", "max_exec_seconds"), -0.1),
            (("budget", "max_exec_seconds"), float("nan")),
            (("budget", "max_cost"), float("inf")),
            (("budget", "max_attempts"), 0),
            (("verification", "minimum_tasks"), 0),
        )
        for path, value in invalid_ranges:
            with self.subTest(path=path, value=value):
                document = _document()
                _set_path(document, path, value)
                with self.assertRaises(ValueError):
                    operator_passport_from_document(document)

    def test_model_enforces_the_same_security_contract(self) -> None:
        invalid_overrides = (
            {"name": "RepairRoute"},
            {"version": "1.0.0-01"},
            {"intent": "short"},
            {"inputs": ("task", "task")},
            {"expected_effects": (EffectScope.EVALUATOR,)},
            {"expected_effects": (EffectScope.TASK_ARTIFACT, EffectScope.TASK_ARTIFACT)},
            {"generator_type": "shell"},
            {"generator_ref": "../operator.txt"},
            {"generator_ref": "https://example.com/operator.txt"},
            {"evidence_ids": ("A" * 64,)},
            {"evidence_ids": (_EVIDENCE_ID, _EVIDENCE_ID)},
            {"lifecycle": "approved"},
            {"signature": 1},
            {"api_version": "evolsa/v2"},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    OperatorPassport(**_passport_values(**override))

        with self.assertRaises(ValueError):
            Budget(max_tokens=True)
        with self.assertRaises(ValueError):
            Budget(max_exec_seconds="4")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            OperatorVerification(baseline="baseline", minimum_tasks=True)
        with self.assertRaises(ValueError):
            OperatorVerification(baseline="baseline", guardrails=("cost", "cost"))

        relative = OperatorPassport(
            **_passport_values(generator_ref="operators/repair-route.txt")
        )
        self.assertEqual(relative.generator_ref, "operators/repair-route.txt")


if __name__ == "__main__":
    unittest.main()
