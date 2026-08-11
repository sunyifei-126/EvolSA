"""Compilation of synthesized drafts into executable L0 operators."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Sequence

from evolsa.core import Applicability, Artifact, Budget, OperatorPassport, SearchNode, TaskState
from evolsa.core.serde import canonical_json, to_primitive
from evolsa.core.hashing import sha256_text
from evolsa.sdk import ModelBackend

from .synthesis import OperatorDraft


OperatorCallback = Callable[[Mapping[str, Any], ModelBackend, Budget], Artifact | str | Mapping[str, Any]]


def _budget_within(requested: Budget, ceiling: Budget) -> bool:
    token_ok = ceiling.max_tokens == 0 or requested.max_tokens <= ceiling.max_tokens
    cost_ok = ceiling.max_cost == 0.0 or requested.max_cost <= ceiling.max_cost
    return (
        token_ok
        and requested.max_exec_seconds <= ceiling.max_exec_seconds
        and requested.max_attempts <= ceiling.max_attempts
        and cost_ok
    )


@dataclass(slots=True)
class CallbackOperator:
    """A generic executable operator backed by a callback or model prompt."""

    passport: OperatorPassport
    prompt_template: str
    callback: OperatorCallback | None = None

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability:
        missing: list[str] = []
        for condition in self.passport.preconditions:
            if not condition.startswith("metadata:"):
                continue
            key = condition.partition(":")[2]
            if not state.metadata.get(key):
                missing.append(condition)
        return Applicability(
            applies=not missing,
            confidence=1.0 if not missing else 0.0,
            reasons=tuple(missing) if missing else ("all machine-checkable preconditions satisfied",),
        )

    def build_context(
        self,
        state: TaskState,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # Evaluator metadata can contain labels, hidden tests, or judge-only
        # diagnostics.  A compiled operator receives the public observation
        # surface, while evidence mining remains a protected host operation.
        def public_observation(parent: SearchNode) -> Mapping[str, Any]:
            observation = parent.observation
            return {
                "status": observation.status.value,
                "output": observation.output,
                "logs": observation.logs,
                "error_signature": observation.error_signature,
                "runtime_seconds": observation.runtime_seconds,
                "token_usage": observation.token_usage,
                "cost": observation.cost,
            }

        return {
            "operator": {
                "id": self.passport.operator_id,
                "intent": self.passport.intent,
                "preconditions": self.passport.preconditions,
                "known_failures": self.passport.known_failures,
            },
            "task": to_primitive(state),
            "parents": [
                {
                    "node_id": parent.node_id,
                    "artifact": to_primitive(parent.artifact),
                    "score": to_primitive(parent.score),
                    "observation": public_observation(parent),
                }
                for parent in parents
            ],
            "experience_view": to_primitive(experience_view),
        }

    def propose(self, context: Mapping[str, Any], model: ModelBackend, budget: Budget) -> Artifact:
        if not _budget_within(budget, self.passport.budget):
            raise ValueError("requested budget exceeds the operator passport ceiling")
        if self.callback is not None:
            result = self.callback(context, model, budget)
        else:
            messages = (
                {"role": "system", "content": self.prompt_template},
                {"role": "user", "content": canonical_json(context)},
            )
            result = model.generate(messages, budget)
        if isinstance(result, Artifact):
            return result
        decoded: Any = result
        if isinstance(result, str):
            try:
                decoded = json.loads(result)
            except json.JSONDecodeError:
                decoded = result
        media_type = "application/json"
        if isinstance(decoded, Mapping) and "content" in decoded:
            raw_content = decoded["content"]
            content = raw_content if isinstance(raw_content, str) else canonical_json(raw_content)
            media_type = str(decoded.get("media_type") or media_type)
        else:
            content = decoded if isinstance(decoded, str) else canonical_json(decoded)
        return Artifact(
            content=content,
            media_type=media_type,
            created_by=self.passport.operator_id,
            metadata={"operator_content_hash": self.passport.content_hash},
        )


class OperatorCompiler:
    """Compile an :class:`OperatorDraft` into an L0 passport and operator."""

    def compile_passport(self, draft: OperatorDraft) -> OperatorPassport:
        return OperatorPassport(
            name=draft.name,
            version=draft.version,
            intent=draft.intent,
            inputs=draft.inputs,
            preconditions=draft.preconditions,
            expected_effects=draft.expected_effects,
            forbidden_effects=draft.forbidden_effects,
            context_policy=draft.context_policy,
            generator_type="prompt_template",
            generator_ref=f"sha256:{sha256_text(draft.prompt_template)}",
            budget=draft.budget,
            verification=draft.verification,
            parents=draft.parents,
            known_failures=draft.known_failures,
            evidence_ids=draft.evidence_ids,
        )

    def compile(self, draft: OperatorDraft, *, callback: OperatorCallback | None = None) -> CallbackOperator:
        return CallbackOperator(
            passport=self.compile_passport(draft),
            prompt_template=draft.prompt_template,
            callback=callback,
        )


__all__ = ["CallbackOperator", "OperatorCallback", "OperatorCompiler"]
