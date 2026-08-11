"""The generation-zero Draft, Improve, Debug, and Crossover operators.

These are deliberately L0 operators: their implementation is a structured
prompt and a model-independent ``ModelBackend`` call.  They do not execute or
score artifacts, and their passports limit effects to the task artifact.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from evolsa.core import (
    Applicability,
    Artifact,
    Budget,
    EffectScope,
    ExecutionStatus,
    OperatorPassport,
    OperatorVerification,
    SearchNode,
    TaskState,
)
from evolsa.core.serde import pretty_json, to_primitive
from evolsa.sdk.protocols import ModelBackend


_FORBIDDEN_EFFECTS = (
    EffectScope.OPERATOR,
    EffectScope.SCHEDULER,
    EffectScope.MEMORY_POLICY,
    EffectScope.EVALUATOR,
    EffectScope.SANDBOX,
    EffectScope.BUDGET,
    EffectScope.PROMOTION_GATE,
    EffectScope.EVENT_LOG,
    EffectScope.RELEASE_PERMISSION,
)


def _bounded_value(requested: float | int, cap: float | int) -> float | int:
    """Apply a positive cap while treating zero as an unspecified limit."""

    if cap == 0:
        return requested
    if requested == 0:
        return cap
    return min(requested, cap)


def _bounded_budget(requested: Budget, cap: Budget) -> Budget:
    return Budget(
        max_tokens=int(_bounded_value(requested.max_tokens, cap.max_tokens)),
        max_exec_seconds=float(min(requested.max_exec_seconds, cap.max_exec_seconds)),
        max_attempts=int(min(requested.max_attempts, cap.max_attempts)),
        max_cost=float(_bounded_value(requested.max_cost, cap.max_cost)),
    )


def _task_payload(state: TaskState) -> Mapping[str, Any]:
    return {
        "task_id": state.task_id,
        "split": state.split,
        "seed": state.seed,
        "specification": state.specification,
        "workspace": state.workspace,
        "metadata": to_primitive(state.metadata),
    }


def _parent_payload(node: SearchNode) -> Mapping[str, Any]:
    return {
        "node_id": node.node_id,
        "operator_id": node.operator_id,
        "generation": node.generation,
        "method_family": node.method_family,
        "plan": node.plan,
        "artifact": {
            "artifact_id": node.artifact.artifact_id,
            "media_type": node.artifact.media_type,
            "content": node.artifact.content,
        },
        "observation": {
            "status": node.observation.status.value,
            "output": node.observation.output,
            "logs": node.observation.logs,
            "error_signature": node.observation.error_signature,
            "runtime_seconds": node.observation.runtime_seconds,
            "token_usage": node.observation.token_usage,
            "cost": node.observation.cost,
        },
        "score": {
            "primary": node.score.primary,
            "direction": node.score.direction.value,
            "valid": node.score.valid,
            "reason": node.score.reason,
        },
    }


class SeedOperator:
    """Shared executable implementation for prompt-template seed operators."""

    operation = "seed"
    parent_count = 0
    system_instruction = "Create a bounded task artifact."

    def __init__(self, passport: OperatorPassport | None = None) -> None:
        self.passport = passport or self.default_passport()

    @classmethod
    def default_passport(cls) -> OperatorPassport:
        raise NotImplementedError

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability:
        del state
        valid = len(parents) == self.parent_count
        return Applicability(
            applies=valid,
            confidence=1.0 if valid else 0.0,
            reasons=(
                f"{self.operation} requires exactly {self.parent_count} distinct parent node(s)",
            ),
        )

    def build_context(
        self,
        state: TaskState,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        applicability = self.applies(state, parents)
        if not applicability.applies:
            raise ValueError(applicability.reasons[0])
        return {
            "schema_version": "evolsa.operator-context/v1",
            "operation": self.operation,
            "operator_id": self.passport.operator_id,
            "context_policy": self.passport.context_policy,
            "task": _task_payload(state),
            "parents": tuple(_parent_payload(parent) for parent in parents),
            "experience": to_primitive(experience_view),
        }

    def propose(
        self,
        context: Mapping[str, Any],
        model: ModelBackend,
        budget: Budget,
    ) -> Artifact:
        if context.get("operation") != self.operation:
            raise ValueError(f"context operation must be {self.operation!r}")
        effective_budget = _bounded_budget(budget, self.passport.budget)
        evidence = {
            "operator_id": self.passport.operator_id,
            "intent": self.passport.intent,
            "allowed_effects": [scope.value for scope in self.passport.expected_effects],
            "forbidden_effects": [scope.value for scope in self.passport.forbidden_effects],
            "effective_budget": to_primitive(effective_budget),
            "context": to_primitive(context),
        }
        messages = (
            {
                "role": "system",
                "content": (
                    "You are an EvolSA L0 task-artifact operator. "
                    f"{self.system_instruction} "
                    "Stay inside the task artifact: never alter the evaluator, sandbox, budget, "
                    "promotion gate, audit log, or release permissions. Return either the artifact "
                    "as plain text, or exactly one JSON object with fields content and media_type. "
                    "Do not wrap the result in Markdown fences."
                ),
            },
            {"role": "user", "content": pretty_json(evidence)},
        )
        raw = model.generate(messages, effective_budget)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("model backend must return a non-empty string")

        content = raw.strip()
        media_type = "text/plain"
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping) and "content" in decoded:
            decoded_content = decoded["content"]
            content = (
                decoded_content
                if isinstance(decoded_content, str)
                else json.dumps(decoded_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            media_type = str(decoded.get("media_type") or "application/json")
        elif decoded is not None:
            media_type = "application/json"
            content = json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            parents = context.get("parents", ())
            if parents and isinstance(parents, Sequence):
                first = parents[0]
                if isinstance(first, Mapping):
                    artifact = first.get("artifact", {})
                    if isinstance(artifact, Mapping):
                        media_type = str(artifact.get("media_type") or media_type)

        parent_artifact_ids: list[str] = []
        for parent in context.get("parents", ()):
            if isinstance(parent, Mapping):
                artifact = parent.get("artifact", {})
                if isinstance(artifact, Mapping) and artifact.get("artifact_id"):
                    parent_artifact_ids.append(str(artifact["artifact_id"]))
        return Artifact(
            content=content,
            media_type=media_type,
            parent_ids=tuple(parent_artifact_ids),
            created_by=self.passport.operator_id,
            metadata={
                "operator_id": self.passport.operator_id,
                "operator_level": "L0",
                "model_id": model.model_id,
                "budget": to_primitive(effective_budget),
            },
        )


class DraftOperator(SeedOperator):
    operation = "draft"
    parent_count = 0
    system_instruction = (
        "Draft an independent first solution from the task specification; do not assume or inherit "
        "any node memory."
    )

    @classmethod
    def default_passport(cls) -> OperatorPassport:
        return OperatorPassport(
            name="draft",
            version="1.0.0",
            intent="create an independent initial task artifact from the specification",
            inputs=("task_spec",),
            preconditions=("task_spec_available",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=_FORBIDDEN_EFFECTS,
            context_policy="draft-no-inherited-memory-v1",
            generator_type="prompt_template",
            generator_ref="builtin/draft-l0",
            budget=Budget(max_tokens=4096, max_exec_seconds=60.0, max_attempts=1),
            verification=OperatorVerification(baseline="seed-artifact", minimum_tasks=4),
        )

    def build_context(
        self,
        state: TaskState,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del experience_view
        # Draft explicitly starts a fresh branch even if a caller has a global board.
        return super().build_context(state, parents, {"policy": "none", "items": ()})


class ImproveOperator(SeedOperator):
    operation = "improve"
    parent_count = 1
    system_instruction = (
        "Improve the selected parent using evidence-backed changes; preserve working behavior and "
        "avoid changes contradicted by ancestor or sibling outcomes."
    )

    @classmethod
    def default_passport(cls) -> OperatorPassport:
        return OperatorPassport(
            name="improve",
            version="1.0.0",
            intent="produce a higher-utility revision of one selected parent artifact",
            inputs=("task_spec", "parent", "experience_view"),
            preconditions=("one_evaluated_parent",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=_FORBIDDEN_EFFECTS,
            context_policy="improve-parent-3-ancestors-3-siblings-v1",
            generator_type="prompt_template",
            generator_ref="builtin/improve-l0",
            budget=Budget(max_tokens=4096, max_exec_seconds=60.0, max_attempts=1),
            verification=OperatorVerification(baseline="draft@1.0.0", minimum_tasks=4),
        )


class DebugOperator(SeedOperator):
    operation = "debug"
    parent_count = 1
    system_instruction = (
        "Repair the selected failing artifact. Diagnose the current error signature first, then "
        "reuse prior fixes for the same failure mode before recent fallback attempts."
    )

    @classmethod
    def default_passport(cls) -> OperatorPassport:
        return OperatorPassport(
            name="debug",
            version="1.0.0",
            intent="diagnose and repair an execution failure in one selected parent artifact",
            inputs=("task_spec", "buggy_parent", "same_error_experience"),
            preconditions=("one_failed_or_buggy_parent",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=_FORBIDDEN_EFFECTS,
            context_policy="debug-same-error-first-v1",
            generator_type="prompt_template",
            generator_ref="builtin/debug-l0",
            budget=Budget(max_tokens=4096, max_exec_seconds=60.0, max_attempts=1),
            verification=OperatorVerification(baseline="improve@1.0.0", minimum_tasks=4),
        )

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability:
        del state
        if len(parents) != 1:
            return Applicability(False, 0.0, ("debug requires exactly one parent node",))
        parent = parents[0]
        failed = bool(parent.observation.error_signature) or parent.observation.status in {
            ExecutionStatus.ERROR,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.INVALID,
        }
        return Applicability(
            failed,
            1.0 if failed else 0.0,
            (("parent has an actionable failure",) if failed else ("parent has no actionable failure",)),
        )


class CrossoverOperator(SeedOperator):
    operation = "crossover"
    parent_count = 2
    system_instruction = (
        "Combine compatible strengths from two distinct parent branches into one coherent artifact; "
        "resolve conflicts explicitly rather than concatenating both artifacts mechanically."
    )

    @classmethod
    def default_passport(cls) -> OperatorPassport:
        return OperatorPassport(
            name="crossover",
            version="1.0.0",
            intent="combine complementary strengths from two distinct parent branches",
            inputs=("task_spec", "parent_a", "parent_b", "branch_experience"),
            preconditions=("two_distinct_evaluated_parents",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=_FORBIDDEN_EFFECTS,
            context_policy="crossover-each-2-ancestors-2-siblings-v1",
            generator_type="prompt_template",
            generator_ref="builtin/crossover-l0",
            budget=Budget(max_tokens=6144, max_exec_seconds=60.0, max_attempts=1),
            verification=OperatorVerification(baseline="improve@1.0.0", minimum_tasks=4),
        )

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability:
        del state
        distinct = len(parents) == 2 and parents[0].node_id != parents[1].node_id
        return Applicability(
            distinct,
            1.0 if distinct else 0.0,
            ("crossover requires exactly two distinct parent nodes sampled without replacement",),
        )


def seed_operators() -> tuple[SeedOperator, ...]:
    """Return a fresh, deterministic generation-zero instruction set."""

    return (DraftOperator(), ImproveOperator(), DebugOperator(), CrossoverOperator())
