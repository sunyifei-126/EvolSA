# EvolSA-Bench Lite

The initial pack is a deterministic CPU-only incident-routing task used to validate the complete
bounded-RSI control flow. It is intentionally small; its purpose is testing provenance, split
discipline, matched budgets, operator discovery, and promotion—not claiming model capability.

Splits have distinct examples:

- `mine`: trace patterns may be mined;
- `validation`: candidate passports may be revised;
- `promotion`: only the immutable gate consumes aggregate paired evidence;
- `sealed`: final cross-generation reporting; never fed back to the foundry.

Each split supports four deterministic seeds. The adapter keeps evaluator examples outside the
`TaskState` passed to operators.
