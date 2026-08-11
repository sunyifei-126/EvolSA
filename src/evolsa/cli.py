"""Dependency-free command line interface for EvolSA.

The CLI is deliberately a thin, auditable layer over the public Python APIs.
Commands print machine-readable JSON on stdout and reserve stderr for errors,
which makes them convenient both for people and for release automation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evolsa import __version__
from evolsa.core import (
    Artifact,
    Budget,
    ExecutionStatus,
    ExperienceCard,
    LabReport,
    MetricDirection,
    PairedOutcome,
    PromotionPolicy,
    PROTECTED_SCOPES,
)
from evolsa.core.serde import pretty_json, to_primitive
from evolsa.core.hashing import sha256_json
from evolsa.foundry import TraceMiner
from evolsa.lab import PromotionGate
from evolsa.registry import LocalRegistry


class CLIError(RuntimeError):
    """A concise user-facing command failure."""


DemoFunction = Callable[[str | os.PathLike[str]], Mapping[str, Any]]


def _official_promotion_policy() -> PromotionPolicy:
    """Return the non-overridable policy bound to the v0.1 reference run."""

    from evolsa.operators import seed_operators

    baseline = next(
        operator.passport for operator in seed_operators() if operator.passport.name == "improve"
    )
    return PromotionPolicy(
        minimum_lcb_lift=0.05,
        valid_rate_tolerance=0.0,
        minimum_cost_adjusted_lift=0.05,
        maximum_regression=0.0,
        minimum_transfer_coverage=4,
        require_matched_budget=True,
        require_matched_conditions=True,
        baseline_id=baseline.operator_id,
        baseline_content_hash=baseline.content_hash,
        required_trials=tuple(
            (f"incident-routing-promotion-{seed}", seed) for seed in range(4)
        ),
        required_split="promotion",
        required_evaluator_id="incident-routing-lite",
        bootstrap_samples=2_000,
        bootstrap_seed=7,
        bootstrap_confidence=0.95,
        cost_penalty=1.0,
    )


def _load_demo_api() -> tuple[DemoFunction, DemoFunction, DemoFunction]:
    """Import the demo lazily so metadata commands stay lightweight."""

    from evolsa.demo.run import inspect_run, replay_run, run_offline_rsi_demo

    return run_offline_rsi_demo, inspect_run, replay_run


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not valid evidence: {value}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = _read_json(path)
    except FileNotFoundError as error:
        raise CLIError(f"required evidence file does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CLIError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CLIError(f"expected a JSON object in {path}")
    return value


def _json_value(path: Path) -> Any:
    try:
        return _read_json(path)
    except FileNotFoundError as error:
        raise CLIError(f"required evidence file does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CLIError(f"cannot read JSON evidence {path}: {error}") from error


def _unwrap(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Accept both direct evidence documents and schema envelope variants."""

    for key in keys:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return value


def _json_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CLIError(f"{label} must be a JSON boolean")
    return value


def _ensure_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise CLIError(f"{label} cannot be a symbolic link: {candidate}")
    if not candidate.is_dir():
        raise CLIError(f"{label} is not a directory: {candidate}")
    return candidate.resolve()


def _ensure_empty_output(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise CLIError(f"output path cannot be a symbolic link: {candidate}")
    if candidate.exists() and not candidate.is_dir():
        raise CLIError(f"output path exists and is not a directory: {candidate}")
    if candidate.is_dir():
        try:
            if next(candidate.iterdir(), None) is not None:
                raise CLIError(f"refusing to overwrite non-empty output directory: {candidate}")
        except OSError as error:
            raise CLIError(f"cannot inspect output directory {candidate}: {error}") from error
    return candidate.resolve()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(pretty_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _emit(value: Any) -> None:
    print(pretty_json(value))


def _budget(value: Any) -> Budget | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CLIError("budget evidence must be an object")
    try:
        return Budget(
            max_tokens=int(value["max_tokens"]),
            max_exec_seconds=float(value["max_exec_seconds"]),
            max_attempts=int(value["max_attempts"]),
            max_cost=float(value.get("max_cost", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid budget evidence: {error}") from error


def _experience_card(value: Mapping[str, Any]) -> ExperienceCard:
    try:
        return ExperienceCard(
            node_id=str(value["node_id"]),
            step_id=int(value["step_id"]),
            operator_id=str(value["operator_id"]),
            parent_node_ids=tuple(str(item) for item in value.get("parent_node_ids", ())),
            generation_id=int(value["generation_id"]),
            score=float(value["score"]),
            utility=float(value["utility"]),
            status=ExecutionStatus(value["status"]),
            error_signature=str(value.get("error_signature", "")),
            runtime_seconds=float(value["runtime_seconds"]),
            token_count=int(value["token_count"]),
            cost=float(value["cost"]),
            method_family=str(value["method_family"]),
            family_count_before=int(value["family_count_before"]),
            delta_vs_parent=float(value["delta_vs_parent"]),
            novelty_score=float(value["novelty_score"]),
            rank=int(value["rank"]),
            current_best=_json_bool(value["current_best"], "experience card current_best"),
            selection_utility=float(value["selection_utility"]),
            plan=str(value.get("plan", "")),
            analysis=str(value.get("analysis", "")),
            method_overview=str(value.get("method_overview", "")),
            parent_comparison_experience=str(value.get("parent_comparison_experience", "")),
            change_tags=tuple(str(item) for item in value.get("change_tags", ())),
            valid=_json_bool(value.get("valid", False), "experience card valid"),
            validity_reason=str(value.get("validity_reason", "")),
            guardrails={str(key): float(item) for key, item in dict(value.get("guardrails", {})).items()},
            artifact_id=str(value.get("artifact_id", "")),
            task_id=str(value.get("task_id", "")),
            split=str(value.get("split", "")),
            seed=int(value.get("seed", -1)),
            evaluator_id=str(value.get("evaluator_id", "")),
            model_id=str(value.get("model_id", "")),
            budget=_budget(value.get("budget")),
            card_id=str(value.get("card_id", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid experience card: {error}") from error


def _read_cards(run_dir: Path) -> tuple[ExperienceCard, ...]:
    raw = _json_value(run_dir / "cards.json")
    if isinstance(raw, Mapping):
        raw = raw.get("cards")
    if not isinstance(raw, list):
        raise CLIError(f"expected an array of cards in {run_dir / 'cards.json'}")
    cards: list[ExperienceCard] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise CLIError(f"experience card at index {index} is not an object")
        cards.append(_experience_card(value))
    return tuple(cards)


def _paired_outcome(value: Mapping[str, Any]) -> PairedOutcome:
    try:
        return PairedOutcome(
            task_id=str(value["task_id"]),
            seed=int(value["seed"]),
            candidate_score=float(value["candidate_score"]),
            baseline_score=float(value["baseline_score"]),
            candidate_valid=_json_bool(value["candidate_valid"], "paired outcome candidate_valid"),
            baseline_valid=_json_bool(value["baseline_valid"], "paired outcome baseline_valid"),
            candidate_cost=float(value["candidate_cost"]),
            baseline_cost=float(value["baseline_cost"]),
            regression=float(value.get("regression", 0.0)),
            direction=MetricDirection(value.get("direction", MetricDirection.HIGHER_IS_BETTER.value)),
            split=str(value.get("split", "")),
            evaluator_id=str(value.get("evaluator_id", "")),
            candidate_parent_id=str(value.get("candidate_parent_id", "")),
            baseline_parent_id=str(value.get("baseline_parent_id", "")),
            candidate_model_id=str(value.get("candidate_model_id", "")),
            baseline_model_id=str(value.get("baseline_model_id", "")),
            candidate_sampling=value.get("candidate_sampling"),
            baseline_sampling=value.get("baseline_sampling"),
            candidate_budget=_budget(value.get("candidate_budget")),
            baseline_budget=_budget(value.get("baseline_budget")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid paired outcome: {error}") from error


def _read_lab_report(run_dir: Path) -> LabReport:
    value = _unwrap(_json_object(run_dir / "lab-report.json"), "report", "lab_report")
    raw_outcomes = value.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise CLIError("lab report outcomes must be an array")
    outcomes: list[PairedOutcome] = []
    for index, outcome in enumerate(raw_outcomes):
        if not isinstance(outcome, Mapping):
            raise CLIError(f"paired outcome at index {index} is not an object")
        outcomes.append(_paired_outcome(outcome))
    try:
        return LabReport(
            operator_id=str(value["operator_id"]),
            baseline_id=str(value["baseline_id"]),
            outcomes=tuple(outcomes),
            mean_lift=float(value["mean_lift"]),
            lcb95=float(value["lcb95"]),
            ucb95=float(value["ucb95"]),
            valid_rate=float(value["valid_rate"]),
            baseline_valid_rate=float(value["baseline_valid_rate"]),
            cost_adjusted_lift=float(value["cost_adjusted_lift"]),
            transfer_coverage=int(value["transfer_coverage"]),
            worst_regression=float(value["worst_regression"]),
            matched_budget=_json_bool(value["matched_budget"], "lab report matched_budget"),
            matched_conditions=_json_bool(
                value.get("matched_conditions", False), "lab report matched_conditions"
            ),
            bootstrap_samples=int(value.get("bootstrap_samples", 10_000)),
            bootstrap_seed=int(value.get("bootstrap_seed", 0)),
            bootstrap_confidence=float(value.get("bootstrap_confidence", 0.95)),
            cost_penalty=float(value.get("cost_penalty", 1.0)),
            report_id=str(value.get("report_id", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid lab report: {error}") from error


def _project_init(output: Path) -> Mapping[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "evolsa.project/v1",
        "mode": "rsi",
        "task_pack": "incident-routing",
        "runs_dir": "runs",
        "registry_dir": "registry",
        "protected_scopes": sorted(scope.value for scope in PROTECTED_SCOPES),
    }
    config_path = output / "evolsa.json"
    _atomic_json(config_path, config)
    return {"initialized": True, "output": str(output), "config": str(config_path), **config}


def _run_demo(args: argparse.Namespace) -> Mapping[str, Any]:
    output = _ensure_empty_output(args.output)
    run_demo, _, _ = _load_demo_api()
    return run_demo(output)


def _replay_if_sealed_run(run_dir: Path) -> Mapping[str, Any] | None:
    """Replay a sealed run before any command consumes its evidence.

    A directory containing run markers is never silently downgraded to an
    unsealed evidence bundle.  Direct card/report bundles remain available for
    local analysis, but commands label their outputs advisory.
    """

    manifest = run_dir / "manifest.json"
    run_markers = (
        manifest,
        run_dir / "run.json",
        run_dir / "events.jsonl",
        run_dir / "events.jsonl.head",
        run_dir / "gate-result.json",
        run_dir / "sealed-results.json",
    )
    if not any(path.exists() for path in run_markers):
        return None
    if manifest.is_symlink() or not manifest.is_file():
        raise CLIError("run evidence is present but manifest.json is missing or unsafe")
    _, _, replay_run = _load_demo_api()
    try:
        receipt = replay_run(run_dir)
    except Exception as error:
        raise CLIError(f"run replay failed: {error}") from error
    if not isinstance(receipt, Mapping):
        raise CLIError("run replay returned a non-object receipt")
    if receipt.get("replay_verified") is not True or receipt.get("semantics_replayed") is not True:
        raise CLIError("run replay did not verify both integrity and semantics")
    return receipt


class _BuiltinSearchModel:
    """Deterministic in-process backend for the built-in search example.

    It consumes the real L0 seed-operator prompt envelope.  This is trusted
    reference code, not an isolation or production evaluation boundary.
    """

    model_id = "evolsa/cli-search-v1"

    def __init__(self, adapter: Any, state: Any) -> None:
        self._adapter = adapter
        self._state = state

    @staticmethod
    def _prompt_context(messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        if not messages:
            raise ValueError("seed operator supplied no model messages")
        try:
            envelope = json.loads(messages[-1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("seed operator supplied an invalid JSON prompt envelope") from error
        if not isinstance(envelope, Mapping) or not isinstance(envelope.get("context"), Mapping):
            raise ValueError("seed operator prompt has no structured context")
        return envelope["context"]

    @staticmethod
    def _valid_parent(context: Mapping[str, Any]) -> Artifact | None:
        from evolsa.demo import parse_policy

        parents = context.get("parents", ())
        if not isinstance(parents, Sequence):
            return None
        for parent in parents:
            if not isinstance(parent, Mapping):
                continue
            raw_artifact = parent.get("artifact")
            if not isinstance(raw_artifact, Mapping):
                continue
            artifact = Artifact(
                content=str(raw_artifact.get("content", "")),
                media_type=str(raw_artifact.get("media_type", "application/json")),
                created_by="cli-search-parent",
            )
            try:
                parse_policy(artifact)
            except ValueError:
                continue
            return artifact
        return None

    def generate(self, messages: Sequence[Mapping[str, str]], budget: Budget) -> str:
        from evolsa.demo import policy_with_normalizer

        if budget.max_attempts < 1 or budget.max_exec_seconds <= 0:
            raise ValueError("search generation budget is exhausted")
        context = self._prompt_context(messages)
        operation = context.get("operation")
        if operation == "draft":
            # A real invalid branch gives Debug an execution failure to repair.
            return json.dumps(
                {"content": "{malformed-policy", "media_type": "application/json"},
                ensure_ascii=False,
                sort_keys=True,
            )
        if operation == "debug":
            artifact = self._adapter.seed_artifact(self._state)
        elif operation in {"improve", "crossover"}:
            parent = self._valid_parent(context) or self._adapter.seed_artifact(self._state)
            normalizer = "strip_punctuation" if operation == "improve" else "hyphen_to_space"
            artifact = policy_with_normalizer(
                parent,
                normalizer,
                created_by=f"{operation}@1.0.0",
            )
        else:
            raise ValueError(f"unsupported built-in search operation: {operation!r}")
        return json.dumps(
            {"content": artifact.content, "media_type": artifact.media_type},
            ensure_ascii=False,
            sort_keys=True,
        )


def _search(args: argparse.Namespace) -> Mapping[str, Any]:
    """Run the fixed, evaluator-backed generation-zero operator search."""

    from evolsa.demo import IncidentRoutingAdapter
    from evolsa.operators import seed_operators
    from evolsa.search import SearchEngine

    adapter = IncidentRoutingAdapter()
    state = adapter.prepare(args.seed, "mine")
    model = _BuiltinSearchModel(adapter, state)
    operators = seed_operators()
    plan = ("draft", "improve", "debug", "crossover")
    budget = Budget(max_tokens=2048, max_exec_seconds=1.0, max_attempts=5)
    result = SearchEngine(
        adapter,
        model,
        operators,
        budget,
        seed=args.seed,
    ).run(state, plan, include_seed=True)
    best = result.best_node
    if best is None:
        raise CLIError("built-in search produced no valid candidate")
    return {
        "schema_version": "evolsa.search-result/v1",
        "mode": "search",
        "task_pack": adapter.name,
        "task_id": state.task_id,
        "seed": args.seed,
        "model_id": model.model_id,
        "operator_ids": [operator.passport.operator_id for operator in operators],
        "operation_plan": list(plan),
        "node_count": len(result.nodes),
        "best_node_id": best.node_id,
        "best_score": best.score.primary,
        "best_artifact": json.loads(best.artifact.content),
        "nodes": [
            {
                "node_id": node.node_id,
                "operator_id": node.operator_id,
                "parent_ids": list(node.parent_ids),
                "status": node.observation.status.value,
                "score": node.score.primary,
                "valid": node.score.valid,
            }
            for node in result.nodes
        ],
        "board": to_primitive(result.board),
        "usage": to_primitive(result.usage),
        "budget_exhausted": result.budget_exhausted,
        "exhaustion_reason": result.exhaustion_reason,
        "trust_boundary": "in-process reference adapter and model backend",
    }


def _inspect(args: argparse.Namespace) -> Mapping[str, Any]:
    run_dir = _ensure_directory(args.run_dir, label="run directory")
    _, inspect_run, _ = _load_demo_api()
    return inspect_run(run_dir)


def _replay(args: argparse.Namespace) -> Mapping[str, Any]:
    run_dir = _ensure_directory(args.run_dir, label="run directory")
    _, _, replay_run = _load_demo_api()
    return replay_run(run_dir)


def _registry_list(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.root).expanduser().resolve()
    records = LocalRegistry(root).list()
    return {
        "root": str(root),
        "count": len(records),
        "operators": [to_primitive(record) for record in records],
    }


def _init(args: argparse.Namespace) -> Mapping[str, Any]:
    return _project_init(_ensure_empty_output(args.output))


def _mine(args: argparse.Namespace) -> Mapping[str, Any]:
    run_dir = _ensure_directory(args.run_dir, label="run directory")
    replay = _replay_if_sealed_run(run_dir)
    cards = _read_cards(run_dir)
    patterns = TraceMiner(
        minimum_support=args.minimum_support,
        minimum_delta=args.minimum_delta,
    ).mine(cards)
    return {
        "run_dir": str(run_dir),
        "evidence_mode": "replayed-sealed-run" if replay is not None else "advisory-unsealed",
        "replay_verified": replay is not None,
        "card_count": len(cards),
        "pattern_count": len(patterns),
        "patterns": [to_primitive(pattern) for pattern in patterns],
    }


def _lab(args: argparse.Namespace) -> Mapping[str, Any]:
    run_dir = _ensure_directory(args.run_dir, label="run directory")
    replay = _replay_if_sealed_run(run_dir)
    report = _read_lab_report(run_dir)
    return {
        "run_dir": str(run_dir),
        "evidence_mode": "replayed-sealed-run" if replay is not None else "advisory-unsealed",
        "replay_verified": replay is not None,
        "report_id": report.report_id,
        "operator_id": report.operator_id,
        "baseline_id": report.baseline_id,
        "mean_lift": report.mean_lift,
        "confidence_interval_95": [report.lcb95, report.ucb95],
        "valid_rate": report.valid_rate,
        "baseline_valid_rate": report.baseline_valid_rate,
        "cost_adjusted_lift": report.cost_adjusted_lift,
        "transfer_coverage": report.transfer_coverage,
        "worst_regression": report.worst_regression,
        "matched_budget": report.matched_budget,
        "matched_conditions": report.matched_conditions,
        "paired_tasks": len(report.outcomes),
    }


def _promote(args: argparse.Namespace) -> Mapping[str, Any]:
    run_dir = _ensure_directory(args.run_dir, label="run directory")
    replay = _replay_if_sealed_run(run_dir)
    report = _read_lab_report(run_dir)
    passport = _unwrap(_json_object(run_dir / "operator-passport.json"), "passport", "operator_passport")
    try:
        operator_id = f"{passport['name']}@{passport['version']}"
    except KeyError as error:
        raise CLIError(f"invalid operator passport: missing {error.args[0]}") from error

    policy = _official_promotion_policy()
    if replay is not None:
        gate = _unwrap(_json_object(run_dir / "gate-result.json"), "gate", "gate_result")
        required = {"decision", "operator_id", "report_id", "reasons", "policy_hash"}
        if set(gate) != required:
            raise CLIError("replayed gate result does not match the v1 contract")
        if gate.get("operator_id") != operator_id or gate.get("report_id") != report.report_id:
            raise CLIError("replayed gate is not bound to the report and operator passport")
        if gate.get("policy_hash") != sha256_json(policy):
            raise CLIError("replayed gate is not bound to the official promotion policy")
        if gate.get("decision") not in {"promote", "shadow", "revise", "reject", "quarantine"}:
            raise CLIError("replayed gate has an invalid decision")
        if not isinstance(gate.get("reasons"), list) or not all(
            isinstance(reason, str) for reason in gate["reasons"]
        ):
            raise CLIError("replayed gate reasons must be an array of strings")
        for field in ("decision", "operator_id", "report_id"):
            if field in replay and replay[field] != gate[field]:
                raise CLIError(f"replay receipt disagrees with the run-bound gate: {field}")
        return {
            "run_dir": str(run_dir),
            "evidence_mode": "replayed-sealed-run",
            "replay_verified": True,
            "decision": gate["decision"],
            "operator_id": gate["operator_id"],
            "report_id": gate["report_id"],
            "reasons": gate["reasons"],
            "policy_hash": gate["policy_hash"],
            "policy": to_primitive(policy),
            "decision_source": "run-bound-gate",
        }

    # Unsealed caller-supplied reports can be explored, but can never produce
    # an official lifecycle decision or promotion receipt.
    advisory = PromotionGate(policy).evaluate(report, operator=operator_id)
    return {
        "run_dir": str(run_dir),
        "evidence_mode": "advisory-unsealed",
        "replay_verified": False,
        "advisory_decision": advisory.decision.value,
        "operator_id": advisory.operator_id,
        "report_id": advisory.report_id,
        "reasons": list(advisory.reasons),
        "policy_hash": advisory.policy_hash,
        "policy": to_primitive(policy),
        "decision_source": "advisory-recomputation-only",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolsa",
        description="Evidence-gated, bounded Recursive Self-Improvement.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("demo", "run the deterministic offline RSI demonstration"),
        ("rsi", "run the complete bounded RSI loop"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", required=True, help="new or empty run output directory")
        command.set_defaults(handler=_run_demo)

    search_command = commands.add_parser(
        "search",
        help="run the built-in evaluator-backed fixed seed-operator search",
    )
    search_command.add_argument("--seed", type=int, default=0, help="deterministic non-negative seed")
    search_command.set_defaults(handler=_search)

    inspect_command = commands.add_parser("inspect", help="summarize a completed run")
    inspect_command.add_argument("run_dir")
    inspect_command.set_defaults(handler=_inspect)

    replay_command = commands.add_parser("replay", help="verify a run's immutable evidence")
    replay_command.add_argument("run_dir")
    replay_command.set_defaults(handler=_replay)

    registry_command = commands.add_parser("registry", help="inspect the local operator registry")
    registry_actions = registry_command.add_subparsers(dest="registry_command", required=True)
    registry_list = registry_actions.add_parser("list", help="list registered operators")
    registry_list.add_argument("--root", required=True, help="registry root")
    registry_list.set_defaults(handler=_registry_list)

    init_command = commands.add_parser("init", help="initialize a bounded-RSI project directory")
    init_command.add_argument("--output", required=True, help="new or empty project directory")
    init_command.set_defaults(handler=_init)

    mine_command = commands.add_parser("mine", help="re-mine operator patterns from run cards")
    mine_command.add_argument("run_dir")
    mine_command.add_argument("--minimum-support", type=int, default=2)
    mine_command.add_argument("--minimum-delta", type=float, default=0.0)
    mine_command.set_defaults(handler=_mine)

    lab_command = commands.add_parser("lab", help="validate and summarize matched-budget lab evidence")
    lab_command.add_argument("run_dir")
    lab_command.set_defaults(handler=_lab)

    promote_command = commands.add_parser(
        "promote",
        help="replay and report the run-bound gate, or advise on an unsealed report",
    )
    promote_command.add_argument("run_dir")
    promote_command.set_defaults(handler=_promote)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return a process status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        if not isinstance(result, Mapping):
            raise CLIError("command returned a non-object result")
        _emit(result)
        return 0
    except CLIError as error:
        parser._print_message(f"error: {error}\n", file=os.sys.stderr)
        return 2
    except Exception as error:  # Keep command failures concise; never swallow process signals.
        parser._print_message(f"error: {error}\n", file=os.sys.stderr)
        return 1


__all__ = ["CLIError", "main"]
