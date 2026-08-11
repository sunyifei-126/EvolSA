from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
import tarfile
from pathlib import Path

from evolsa.operators import seed_operators
from evolsa.schemas import (
    OPERATOR_PASSPORT_SCHEMA,
    load_operator_passport_schema,
    operator_passport_document,
    operator_passport_from_document,
    operator_passport_schema_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


def has_setuptools_build_backend() -> bool:
    """Return whether source-build QA can run in the current interpreter."""

    try:
        return (
            importlib.util.find_spec("setuptools") is not None
            and importlib.util.find_spec("setuptools.build_meta") is not None
        )
    except (ImportError, ModuleNotFoundError):
        return False


def copy_project(destination: Path) -> None:
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", ".evolsa", "__pycache__", "*.egg-info", "build", "dist"
        ),
    )


class SchemaResourceTests(unittest.TestCase):
    def test_packaged_schema_matches_repository_contract_byte_for_byte(self) -> None:
        repository_bytes = (ROOT / "schemas" / OPERATOR_PASSPORT_SCHEMA).read_bytes()
        self.assertEqual(operator_passport_schema_bytes(), repository_bytes)

    def test_schema_loader_returns_an_independent_parsed_document(self) -> None:
        first = load_operator_passport_schema()
        second = load_operator_passport_schema()

        self.assertEqual(first["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(first["properties"]["api_version"]["const"], "evolsa/v1")
        first["title"] = "local annotation"
        self.assertEqual(second["title"], "EvolSA Operator Passport")

    def test_packaged_bytes_are_valid_utf8_json(self) -> None:
        decoded = json.loads(operator_passport_schema_bytes().decode("utf-8"))
        self.assertFalse(decoded["additionalProperties"])
        self.assertIn("forbidden_effects", decoded["required"])


class ReleaseAuditTests(unittest.TestCase):
    def test_static_seed_passports_are_runtime_round_trips(self) -> None:
        runtime = {operator.passport.name: operator.passport for operator in seed_operators()}
        paths = sorted((ROOT / "operators" / "core").glob("*.json"))
        self.assertEqual({path.stem for path in paths}, set(runtime))
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(operator_passport_from_document(document), runtime[path.stem])
            self.assertEqual(document, operator_passport_document(runtime[path.stem]))

    def test_release_audit_passes(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/check_release.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("packaged schema sha256=", completed.stdout)

    def test_release_audit_rejects_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolsa-audit-") as temporary:
            project = Path(temporary) / "project"
            copy_project(project)
            packaged_schema = project / "src/evolsa/schemas/operator-passport.schema.json"
            packaged_schema.write_bytes(packaged_schema.read_bytes() + b"\n")

            completed = subprocess.run(
                [sys.executable, "scripts/check_release.py"],
                cwd=project,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("packaged operator passport schemas differ", completed.stderr)

    def test_built_wheel_contains_schema_resource(self) -> None:
        """Exercise setuptools' wheel selection rather than only source imports."""

        if not has_setuptools_build_backend():
            self.skipTest("source-build QA requires the setuptools build backend")

        with tempfile.TemporaryDirectory(prefix="evolsa-wheel-") as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            output = temporary_root / "dist"
            copy_project(project)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(output),
                ],
                cwd=project,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            wheels = list(output.glob("evolsa_rsi-*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
                resource_name = f"evolsa/schemas/{OPERATOR_PASSPORT_SCHEMA}"
                self.assertIn(resource_name, names)
                self.assertEqual(
                    archive.read(resource_name),
                    (ROOT / "schemas" / OPERATOR_PASSPORT_SCHEMA).read_bytes(),
                )

    def test_source_distribution_contains_governance_and_release_tooling(self) -> None:
        if not has_setuptools_build_backend():
            self.skipTest("source-build QA requires the setuptools build backend")
        with tempfile.TemporaryDirectory(prefix="evolsa-sdist-") as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            output = temporary_root / "dist"
            copy_project(project)
            sdist = subprocess.run(
                [sys.executable, "setup.py", "sdist", "--dist-dir", str(output)],
                cwd=project,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(sdist.returncode, 0, sdist.stdout + sdist.stderr)
            archives = list(output.glob("evolsa_rsi-*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                names = {Path(name).parts[-1] for name in archive.getnames()}
                for required in (
                    "SECURITY.md",
                    "GOVERNANCE.md",
                    "CONTRIBUTING.md",
                    "CODE_OF_CONDUCT.md",
                    "Makefile",
                    "check_release.py",
                    "release.yml",
                ):
                    self.assertIn(required, names)
                extracted = temporary_root / "extracted"
                archive.extractall(extracted, filter="data")
            roots = [path for path in extracted.iterdir() if path.is_dir()]
            self.assertEqual(len(roots), 1)
            audit = subprocess.run(
                [sys.executable, "scripts/check_release.py"],
                cwd=roots[0],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(audit.returncode, 0, audit.stdout + audit.stderr)


if __name__ == "__main__":
    unittest.main()
