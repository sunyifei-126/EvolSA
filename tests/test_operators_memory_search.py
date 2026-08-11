from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace

from evolsa.core import (
    Artifact,
    Budget,
    ExecutionStatus,
    MetricDirection,
    Observation,
    ScoreCard,
    SearchNode,
    TaskState,
)
from evolsa.memory import ExperienceStore, OperationConditionedRetriever
from evolsa.foundry import CallbackOperator
from evolsa.operators import (
    CrossoverOperator,
    DebugOperator,
    DraftOperator,
    ImproveOperator,
    bind_operation_family,
)
from evolsa.search import QualityProgressNoveltyScheduler, SearchDAG, SearchEngine


def node(
    node_id: str,
    score: float,
    *,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    parents: tuple[str, ...] = (),
    generation: int = 0,
    family: str = "family-a",
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    error: str = "",
    operator: str = "improve@1.0.0",
) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        artifact=Artifact(content=f"artifact:{node_id}", media_type="text/plain"),
        operator_id=operator,
        parent_ids=parents,
        generation=generation,
        observation=Observation(
            status=status,
            logs=f"logs:{node_id}",
            error_signature=error,
            runtime_seconds=float(generation + 1),
            token_usage=10 + generation,
            cost=0.01,
        ),
        score=ScoreCard(primary=score, direction=direction, valid=status is ExecutionStatus.SUCCESS),
        method_family=family,
        plan=f"plan:{node_id}",
    )


class RecordingModel:
    model_id = "recording-model"

    def __init__(self) -> None:
        self.calls: list[tuple[object, Budget]] = []

    def generate(self, messages: object, budget: Budget) -> str:
        self.calls.append((messages, budget))
        return json.dumps({"content": "generated", "media_type": "text/plain"})


class SimpleAdapter:
    name = "simple"

    def prepare(self, seed: int, split: str = "mine") -> TaskState:
        return TaskState("task", split, seed, "produce a value")

    def seed_artifact(self, state: TaskState) -> Artifact:
        return Artifact("seed", media_type="text/plain", metadata={"method_family": "seed"})

    def execute(self, state: TaskState, artifact: Artifact, budget: Budget) -> Observation:
        return Observation(ExecutionStatus.SUCCESS, output=artifact.content, runtime_seconds=1, token_usage=2)

    def score(self, state: TaskState, observation: Observation) -> ScoreCard:
        return ScoreCard(float(len(observation.output)))


class OverspendAdapter(SimpleAdapter):
    def execute(self, state: TaskState, artifact: Artifact, budget: Budget) -> Observation:
        del state, artifact, budget
        return Observation(
            ExecutionStatus.SUCCESS,
            output="reported-success",
            runtime_seconds=99.0,
            token_usage=999,
            cost=99.0,
        )

    def score(self, state: TaskState, observation: Observation) -> ScoreCard:
        del state
        return ScoreCard(
            1.0 if observation.status is ExecutionStatus.SUCCESS else 0.0,
            valid=observation.status is ExecutionStatus.SUCCESS,
        )


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = TaskState("task", "mine", 7, "return structured output")

    def test_seed_operator_arity_context_and_budget_are_explicit(self) -> None:
        parent = node("p", 1.0)
        model = RecordingModel()
        improve = ImproveOperator()
        context = improve.build_context(self.state, (parent,), {"ancestors": ()})
        artifact = improve.propose(context, model, Budget(max_tokens=99999, max_exec_seconds=120, max_attempts=4))

        self.assertEqual(context["operation"], "improve")
        self.assertEqual(context["parents"][0]["node_id"], "p")
        self.assertEqual(model.calls[0][1].max_tokens, improve.passport.budget.max_tokens)
        self.assertEqual(model.calls[0][1].max_attempts, 1)
        self.assertEqual(artifact.parent_ids, (parent.artifact.artifact_id,))
        self.assertEqual(artifact.created_by, "improve@1.0.0")
        with self.assertRaises(ValueError):
            improve.build_context(self.state, (), {})

    def test_draft_forgets_memory_debug_requires_failure_and_crossover_is_distinct(self) -> None:
        draft_context = DraftOperator().build_context(self.state, (), {"secret_history": "must not leak"})
        self.assertEqual(draft_context["experience"], {"items": [], "policy": "none"})
        success = node("ok", 1.0)
        failed = node("bad", 0.0, status=ExecutionStatus.ERROR, error="ValueError:x")
        self.assertFalse(DebugOperator().applies(self.state, (success,)).applies)
        self.assertTrue(DebugOperator().applies(self.state, (failed,)).applies)
        self.assertFalse(CrossoverOperator().applies(self.state, (success, success)).applies)


class SchedulerTests(unittest.TestCase):
    def test_quality_and_progress_are_direction_aware(self) -> None:
        high_parent = node("hp", 1.0)
        high_child = node("hc", 3.0, parents=("hp",), generation=1)
        high_other = node("ho", 2.0, family="family-b")
        high_scores = QualityProgressNoveltyScheduler().score(
            (high_child, high_other), population=(high_parent, high_child, high_other)
        )
        by_id = {item.node_id: item for item in high_scores}
        self.assertEqual(by_id["hc"].quality, 1.0)
        self.assertEqual(by_id["hc"].progress, 1.0)

        low_parent = node("lp", 5.0, direction=MetricDirection.LOWER_IS_BETTER)
        low_child = node(
            "lc", 3.0, direction=MetricDirection.LOWER_IS_BETTER, parents=("lp",), generation=1
        )
        low_other = node("lo", 4.0, direction=MetricDirection.LOWER_IS_BETTER, family="family-b")
        low_scores = QualityProgressNoveltyScheduler().score(
            (low_child, low_other), population=(low_parent, low_child, low_other)
        )
        by_id = {item.node_id: item for item in low_scores}
        self.assertEqual(by_id["lc"].quality, 1.0)
        self.assertEqual(by_id["lc"].progress, 1.0)

    def test_equal_scores_split_probability_evenly(self) -> None:
        candidates = tuple(node(f"n{index}", 2.0) for index in range(3))
        scores = QualityProgressNoveltyScheduler(temperature=0.25).score(candidates)
        for score in scores:
            self.assertEqual(score.quality, 0.5)
            self.assertAlmostEqual(score.probability, 1 / 3)

    def test_novelty_formula_and_reproducible_crossover_without_replacement(self) -> None:
        candidates = tuple(node(f"n{index}", float(index)) for index in range(4))
        scores = QualityProgressNoveltyScheduler().score(candidates, family_counts={"family-a": 3})
        self.assertTrue(all(math.isclose(score.novelty, 0.5) for score in scores))

        first = QualityProgressNoveltyScheduler(seed=91).select("crossover", candidates)
        second = QualityProgressNoveltyScheduler(seed=91).select("crossover", candidates)
        self.assertEqual(tuple(item.node_id for item in first), tuple(item.node_id for item in second))
        self.assertEqual(len({item.node_id for item in first}), 2)
        with self.assertRaises(ValueError):
            QualityProgressNoveltyScheduler().select("crossover", candidates[:1])


class MemoryTests(unittest.TestCase):
    def _store(self) -> ExperienceStore:
        store = ExperienceStore()
        store.add(node("root", 1.0, family="root"))
        store.add(node("a1", 2.0, parents=("root",), generation=1))
        store.add(node("a2", 3.0, parents=("a1",), generation=2))
        store.add(node("parent", 4.0, parents=("a2",), generation=3))
        for index, score in enumerate((3.5, 4.5, 2.5, 5.0), start=1):
            store.add(node(f"sib{index}", score, parents=("a2",), generation=3, family=f"sib-{index}"))
        return store

    def test_cards_and_board_are_direction_aware(self) -> None:
        store = ExperienceStore()
        store.add(node("root", 5.0, direction=MetricDirection.LOWER_IS_BETTER))
        card = store.add(
            node(
                "child",
                3.0,
                direction=MetricDirection.LOWER_IS_BETTER,
                parents=("root",),
                generation=1,
            )
        )
        self.assertEqual(card.delta_vs_parent, 2.0)
        self.assertEqual(store.board.best_node_id, "child")
        self.assertEqual(store.board.best_score, 3.0)

    def test_improve_retrieval_caps_ancestors_and_siblings(self) -> None:
        store = self._store()
        view = OperationConditionedRetriever(store).retrieve("improve", ("parent",))
        self.assertEqual([card.node_id for card in view["ancestors"]], ["a2", "a1", "root"])
        self.assertEqual(len(view["siblings"]), 3)
        self.assertNotIn("sib3", {card.node_id for card in view["siblings"]})
        self.assertEqual(view["parent"].node_id, "parent")

    def test_draft_and_crossover_retrieval_boundaries(self) -> None:
        store = self._store()
        retriever = OperationConditionedRetriever(store)
        self.assertEqual(retriever.retrieve("draft", ("not-even-a-node",))["memory"], ())
        with self.assertRaises(ValueError):
            retriever.retrieve("crossover", ("parent", "parent"))
        view = retriever.retrieve("crossover", ("parent", "sib1"))
        self.assertEqual(len(view["branches"]), 2)
        self.assertTrue(all(len(branch["ancestors"]) <= 2 for branch in view["branches"]))
        self.assertTrue(all(len(branch["siblings"]) <= 2 for branch in view["branches"]))

    def test_debug_prefers_same_error_then_recent_fallback(self) -> None:
        store = ExperienceStore()
        store.add(node("err-old", 0, status=ExecutionStatus.ERROR, error="E:same"))
        store.add(node("other", 0, status=ExecutionStatus.ERROR, error="E:other"))
        store.add(node("ok", 1))
        store.add(node("err-new", 0, status=ExecutionStatus.ERROR, error="E:same"))
        view = OperationConditionedRetriever(store).retrieve("debug", ("err-new",))
        self.assertEqual(view["related"][0].node_id, "err-old")
        self.assertEqual(len(view["related"]), 3)
        self.assertEqual(view["repeated_error_count"], 2)


class DagAndEngineTests(unittest.TestCase):
    def test_dag_rejects_missing_parent_and_tracks_topology(self) -> None:
        dag = SearchDAG()
        with self.assertRaises(ValueError):
            dag.add(node("child", 2, parents=("missing",), generation=1))
        dag.add(node("root", 1))
        dag.add(node("left", 2, parents=("root",), generation=1))
        dag.add(node("right", 3, parents=("root",), generation=1))
        self.assertEqual({item.node_id for item in dag.children("root")}, {"left", "right"})
        self.assertEqual(dag.best_node.node_id, "right")
        self.assertEqual({item.node_id for item in dag.siblings("left")}, {"right"})

    def test_engine_stops_at_fixed_attempt_budget(self) -> None:
        adapter = SimpleAdapter()
        state = adapter.prepare(3)
        engine = SearchEngine(
            adapter,
            RecordingModel(),
            (DraftOperator(),),
            Budget(max_tokens=100, max_exec_seconds=10, max_attempts=2),
        )
        result = engine.run(state, ("draft", "draft", "draft"))
        self.assertEqual(result.usage.attempts, 2)
        self.assertEqual(len(result.nodes), 2)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.exhaustion_reason, "max_attempts")

    def test_reported_budget_overrun_is_ineligible_evidence(self) -> None:
        adapter = OverspendAdapter()
        state = adapter.prepare(3)
        engine = SearchEngine(
            adapter,
            RecordingModel(),
            (DraftOperator(),),
            Budget(max_tokens=1, max_exec_seconds=1.0, max_attempts=1, max_cost=1.0),
        )
        result = engine.run(state, ("draft",), include_seed=False)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0].observation.status, ExecutionStatus.BUDGET_EXHAUSTED)
        self.assertFalse(result.cards[0].valid)
        self.assertEqual(engine.dag.eligible_nodes, ())

    def test_evolved_operator_uses_explicit_family_not_passport_name(self) -> None:
        adapter = SimpleAdapter()
        state = adapter.prepare(17)
        passport = replace(
            ImproveOperator().passport,
            name="normalize-punctuation",
            version="0.1.0",
        )
        installed = CallbackOperator(
            passport=passport,
            prompt_template="Normalize punctuation while preserving the answer.",
        )
        with self.assertRaisesRegex(ValueError, "bind_operation_family"):
            SearchEngine(
                adapter,
                RecordingModel(),
                (installed,),
                Budget(max_tokens=100, max_exec_seconds=10, max_attempts=2),
            ).run(state, ("normalize-punctuation",))
        evolved = bind_operation_family(installed, "improve")
        engine = SearchEngine(
            adapter,
            RecordingModel(),
            (evolved,),
            # The run has multiple attempts while the installed operator keeps
            # its one-attempt passport ceiling. SearchEngine must pass the
            # effective per-step intersection to CallbackOperator.
            Budget(max_tokens=100, max_exec_seconds=10, max_attempts=3),
        )

        result = engine.run(state, ("normalize-punctuation",))

        self.assertEqual(len(result.nodes), 2)
        evolved_node = result.nodes[-1]
        self.assertEqual(evolved.operation_family, "improve")
        self.assertEqual(evolved.parent_count, 1)
        self.assertEqual(evolved_node.operator_id, "normalize-punctuation@0.1.0")
        self.assertEqual(evolved_node.parent_ids, (result.nodes[0].node_id,))
        self.assertEqual(evolved_node.budget.max_attempts, 1)
        self.assertIn(evolved_node, engine.dag.eligible_nodes)


if __name__ == "__main__":
    unittest.main()
