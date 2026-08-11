from __future__ import annotations

import unittest

from evolsa.core import Budget, ExecutionStatus
from evolsa.demo import (
    IncidentRoutingAdapter,
    infer_normalizer_from_feedback,
    policy_with_normalizer,
)


class IncidentRoutingAdapterTests(unittest.TestCase):
    def test_mining_feedback_discovers_the_transferable_normalizer(self) -> None:
        adapter = IncidentRoutingAdapter()
        state = adapter.prepare(0, "mine")
        seed = adapter.seed_artifact(state)
        observation = adapter.execute(state, seed, Budget())
        self.assertEqual(
            infer_normalizer_from_feedback(seed, observation.metadata["misses"]),
            "hyphen_to_space",
        )

    def test_task_inputs_are_disjoint_across_all_protocol_splits(self) -> None:
        adapter = IncidentRoutingAdapter()
        by_split = {
            split: {
                digest
                for seed in range(4)
                for digest in adapter.case_fingerprints(seed, split)
            }
            for split in ("mine", "validation", "promotion", "sealed")
        }
        splits = tuple(by_split)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                self.assertTrue(by_split[left].isdisjoint(by_split[right]), (left, right))

    def test_hyphen_normalizer_transfers_across_splits(self) -> None:
        adapter = IncidentRoutingAdapter()
        for split in ("mine", "validation", "promotion", "sealed"):
            state = adapter.prepare(1, split)
            seed = adapter.seed_artifact(state)
            candidate = policy_with_normalizer(seed, "hyphen_to_space", created_by="test")
            baseline_score = adapter.score(state, adapter.execute(state, seed, Budget())).primary
            candidate_score = adapter.score(state, adapter.execute(state, candidate, Budget())).primary
            self.assertGreater(candidate_score, baseline_score)
            self.assertEqual(candidate_score, 1.0)

    def test_invalid_policy_fails_closed(self) -> None:
        from evolsa.core import Artifact

        adapter = IncidentRoutingAdapter()
        state = adapter.prepare(0)
        observation = adapter.execute(state, Artifact(content='{"protected": true}'), Budget())
        self.assertEqual(observation.status, ExecutionStatus.INVALID)
        self.assertFalse(adapter.score(state, observation).valid)


if __name__ == "__main__":
    unittest.main()
