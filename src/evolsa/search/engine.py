"""A minimal finite-budget engine joining operators, memory, and DAG search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from evolsa.core import (
    Artifact,
    Budget,
    BudgetUsage,
    ExecutionStatus,
    ExperienceBoard,
    ExperienceCard,
    SearchNode,
    TaskState,
)
from evolsa.core.hashing import sha256_json
from evolsa.memory import ExperienceStore, OperationConditionedRetriever
from evolsa.operators.runtime import operation_family_for
from evolsa.sdk.protocols import ModelBackend, Operator, TaskAdapter

from .dag import SearchDAG
from .scheduler import QualityProgressNoveltyScheduler


class BudgetExhaustedError(RuntimeError):
    """Raised when an operation would exceed the run's fixed attempt budget."""


def _optional_limit_intersection(first: float | int, second: float | int) -> float | int:
    """Intersect limits whose zero value means that meter is not configured."""

    if first == 0:
        return second
    if second == 0:
        return first
    return min(first, second)


def _operator_step_budget(remaining: Budget, ceiling: Budget) -> Budget:
    """Apply an operator passport ceiling to the remaining run budget."""

    return Budget(
        max_tokens=int(_optional_limit_intersection(remaining.max_tokens, ceiling.max_tokens)),
        max_exec_seconds=min(remaining.max_exec_seconds, ceiling.max_exec_seconds),
        max_attempts=min(remaining.max_attempts, ceiling.max_attempts),
        max_cost=float(_optional_limit_intersection(remaining.max_cost, ceiling.max_cost)),
    )


@dataclass(frozen=True, slots=True)
class SearchResult:
    nodes: tuple[SearchNode, ...]
    cards: tuple[ExperienceCard, ...]
    board: ExperienceBoard
    usage: BudgetUsage
    budget_exhausted: bool
    exhaustion_reason: str = ""

    @property
    def best_node(self) -> SearchNode | None:
        valid = [
            node
            for node in self.nodes
            if node.node_id == self.board.best_node_id
        ]
        return valid[0] if valid else None


class SearchEngine:
    """Execute a caller-supplied operation plan under immutable run limits.

    This base engine intentionally does not invent a stopping policy or mutate
    operators.  The caller supplies the sequence of operations; the engine
    enforces the budget, selects distinct parents, retrieves bounded memory,
    and records every evaluated artifact.
    """

    def __init__(
        self,
        adapter: TaskAdapter,
        model: ModelBackend,
        operators: Iterable[Operator],
        budget: Budget,
        *,
        scheduler: QualityProgressNoveltyScheduler | None = None,
        dag: SearchDAG | None = None,
        experiences: ExperienceStore | None = None,
        seed: int = 0,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.budget = budget
        self.scheduler = scheduler or QualityProgressNoveltyScheduler(seed=seed)
        self.dag = dag if dag is not None else SearchDAG()
        self.experiences = experiences if experiences is not None else ExperienceStore()
        if set(self.dag.nodes) != set(self.experiences.nodes):
            raise ValueError("preloaded DAG and ExperienceStore must contain the same node IDs")
        self.retriever = OperationConditionedRetriever(self.experiences)
        self._operators: dict[str, Operator] = {}
        for operator in operators:
            self._operators[operator.passport.operator_id] = operator
            self._operators.setdefault(operator.passport.name, operator)
        if not self._operators:
            raise ValueError("SearchEngine requires at least one operator")
        self._tokens = 0
        self._seconds = 0.0
        self._attempts = 0
        self._cost = 0.0

    @property
    def usage(self) -> BudgetUsage:
        return BudgetUsage(
            tokens=self._tokens,
            exec_seconds=self._seconds,
            attempts=self._attempts,
            cost=self._cost,
        )

    @property
    def exhaustion_reason(self) -> str:
        if self._attempts >= self.budget.max_attempts:
            return "max_attempts"
        if self._seconds >= self.budget.max_exec_seconds:
            return "max_exec_seconds"
        if self.budget.max_tokens and self._tokens >= self.budget.max_tokens:
            return "max_tokens"
        if self.budget.max_cost and self._cost >= self.budget.max_cost:
            return "max_cost"
        return ""

    @property
    def budget_exhausted(self) -> bool:
        return bool(self.exhaustion_reason)

    def _remaining_budget(self) -> Budget:
        if self.budget_exhausted:
            raise BudgetExhaustedError(f"search budget exhausted: {self.exhaustion_reason}")
        return Budget(
            max_tokens=(
                max(0, self.budget.max_tokens - self._tokens) if self.budget.max_tokens else 0
            ),
            max_exec_seconds=max(0.0, self.budget.max_exec_seconds - self._seconds),
            max_attempts=max(1, self.budget.max_attempts - self._attempts),
            max_cost=(max(0.0, self.budget.max_cost - self._cost) if self.budget.max_cost else 0.0),
        )

    def _reserve_attempt(self) -> Budget:
        remaining = self._remaining_budget()
        self._attempts += 1
        return remaining

    def _record_observation(self, observation: object) -> None:
        self._tokens += int(getattr(observation, "token_usage"))
        self._seconds += float(getattr(observation, "runtime_seconds"))
        self._cost += float(getattr(observation, "cost"))

    def _evaluate(
        self,
        state: TaskState,
        artifact: Artifact,
        run_budget: Budget,
        *,
        operator_id: str,
        parent_ids: tuple[str, ...],
        generation: int,
        method_family: str,
        plan: str,
    ) -> SearchNode:
        observation = self.adapter.execute(state, artifact, run_budget)
        self._record_observation(observation)
        budget_violations: list[str] = []
        if run_budget.max_tokens and observation.token_usage > run_budget.max_tokens:
            budget_violations.append("tokens")
        if observation.runtime_seconds > run_budget.max_exec_seconds + 1e-12:
            budget_violations.append("exec_seconds")
        if run_budget.max_cost and observation.cost > run_budget.max_cost + 1e-12:
            budget_violations.append("cost")
        if budget_violations:
            observation = replace(
                observation,
                status=ExecutionStatus.BUDGET_EXHAUSTED,
                error_signature="budget-overrun:" + ",".join(budget_violations),
                metadata={
                    **dict(observation.metadata),
                    "reported_status": observation.status.value,
                    "budget_violations": tuple(budget_violations),
                    "budget_ceiling": {
                        "max_tokens": run_budget.max_tokens,
                        "max_exec_seconds": run_budget.max_exec_seconds,
                        "max_cost": run_budget.max_cost,
                    },
                },
            )
        score = self.adapter.score(state, observation)
        evaluator_id = str(getattr(self.adapter, "evaluator_id", getattr(self.adapter, "name", type(self.adapter).__name__)))
        model_id = str(getattr(self.model, "model_id", type(self.model).__name__))
        node_id = sha256_json(
            {
                "task_id": state.task_id,
                "split": state.split,
                "seed": state.seed,
                "artifact_id": artifact.artifact_id,
                "operator_id": operator_id,
                "parent_ids": parent_ids,
                "generation": generation,
                "attempt": self._attempts,
                "observation": observation,
                "score": score,
                "budget": run_budget,
                "evaluator_id": evaluator_id,
                "model_id": model_id,
            }
        )
        node = SearchNode(
            node_id=node_id,
            artifact=artifact,
            operator_id=operator_id,
            parent_ids=parent_ids,
            generation=generation,
            observation=observation,
            score=score,
            method_family=method_family,
            plan=plan,
            task_id=state.task_id,
            split=state.split,
            seed=state.seed,
            evaluator_id=evaluator_id,
            model_id=model_id,
            budget=run_budget,
        )
        self.dag.add(node)
        self.experiences.add(node)
        return node

    def add_seed(self, state: TaskState, artifact: Artifact | None = None) -> SearchNode:
        run_budget = self._reserve_attempt()
        artifact = artifact or self.adapter.seed_artifact(state)
        metadata = artifact.metadata
        return self._evaluate(
            state,
            artifact,
            run_budget,
            operator_id="seed",
            parent_ids=(),
            generation=0,
            method_family=str(metadata.get("method_family", "seed")),
            plan=str(metadata.get("plan", "seed artifact")),
        )

    def _resolve_operator(self, operator: str | Operator) -> Operator:
        if not isinstance(operator, str):
            return operator
        try:
            return self._operators[operator]
        except KeyError as exc:
            raise KeyError(f"unknown search operator: {operator}") from exc

    def step(self, state: TaskState, operator: str | Operator) -> SearchNode:
        selected_operator = self._resolve_operator(operator)
        operation = operation_family_for(selected_operator)
        if operation == "draft":
            parents: tuple[SearchNode, ...] = ()
        else:
            candidates = self.dag.failed_nodes if operation == "debug" else self.dag.eligible_nodes
            parents = self.scheduler.select(
                selected_operator,
                candidates,
                population=tuple(self.dag.nodes.values()),
            )
            if self.scheduler.last_trace is not None:
                self.experiences.set_parent_selection_weights(
                    {score.node_id: score.probability for score in self.scheduler.last_trace.candidates}
                )
        applicability = selected_operator.applies(state, parents)
        if not applicability.applies:
            raise ValueError("; ".join(applicability.reasons))
        experience = self.retriever.retrieve(selected_operator, parents)
        context = selected_operator.build_context(state, parents, experience)
        remaining_budget = self._reserve_attempt()
        run_budget = _operator_step_budget(remaining_budget, selected_operator.passport.budget)
        artifact = selected_operator.propose(context, self.model, run_budget)
        generation = 0 if not parents else 1 + max(parent.generation for parent in parents)
        method_family = str(
            artifact.metadata.get(
                "method_family",
                "+".join(dict.fromkeys(parent.method_family for parent in parents)) or operation,
            )
        )
        return self._evaluate(
            state,
            artifact,
            run_budget,
            operator_id=selected_operator.passport.operator_id,
            parent_ids=tuple(parent.node_id for parent in parents),
            generation=generation,
            method_family=method_family,
            plan=str(artifact.metadata.get("plan", selected_operator.passport.intent)),
        )

    def run(
        self,
        state: TaskState,
        operation_plan: Sequence[str | Operator],
        *,
        include_seed: bool = True,
        seed_artifact: Artifact | None = None,
    ) -> SearchResult:
        if include_seed and not self.dag.nodes and not self.budget_exhausted:
            self.add_seed(state, seed_artifact)
        for operator in operation_plan:
            if self.budget_exhausted:
                break
            self.step(state, operator)
        return SearchResult(
            nodes=tuple(self.dag.nodes.values()),
            cards=tuple(self.experiences.cards.values()),
            board=self.experiences.board,
            usage=self.usage,
            budget_exhausted=self.budget_exhausted,
            exhaustion_reason=self.exhaustion_reason,
        )
