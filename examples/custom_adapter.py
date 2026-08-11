"""Minimal deterministic TaskAdapter example."""

from evolsa.core import Artifact, Budget, ExecutionStatus, Observation, ScoreCard, TaskState


class ExactMatchAdapter:
    name = "exact-match"

    def prepare(self, seed: int, split: str = "mine") -> TaskState:
        return TaskState(f"exact-{split}-{seed}", split, seed, "Produce the token EVOLSA")

    def seed_artifact(self, state: TaskState) -> Artifact:
        del state
        return Artifact("evol")

    def execute(self, state: TaskState, artifact: Artifact, budget: Budget) -> Observation:
        del state, budget
        return Observation(ExecutionStatus.SUCCESS, output=artifact.content)

    def score(self, state: TaskState, observation: Observation) -> ScoreCard:
        del state
        return ScoreCard(primary=float(observation.output == "EVOLSA"))
