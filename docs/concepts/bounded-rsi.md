# Bounded Recursive Self-Improvement

EvolSA uses a falsifiable definition of bounded RSI. Generation `g` has an improvement genome
`G(g) = (O(g), π(g), M(g))`: operators, a scheduling policy, and a memory policy. A run provides
recursive-improvement evidence only when:

1. generation `g` proposes a change `ΔG` to that genome;
2. the change is evaluated outside the genome on held-out tasks under a matched budget;
3. a host-controlled promotion policy installs the change as generation `g+1`;
4. `g+1` improves the rate of improvement on sealed tasks, not merely the final score;
5. lineage and raw evidence attribute the change to a specific passport.

Task-local reflection, best-of-N sampling, and a better final artifact are useful but do not meet
this definition on their own.

## Boundary

The conceptual mutable surface includes task artifacts, prompts and tool graphs, operator
applicability, scheduler parameters, and memory retrieval/compression. Evaluator implementation,
sealed-task allocation, sandbox permissions, hard budgets, append-only events, minimum promotion
thresholds, signatures, and release authority are protected by the protocol.

This boundary is a design constraint, not a claim that an in-process Python object is immutable
against an operating-system adversary. Production systems must enforce the boundary outside
the agent process. In v0.1, Python callbacks and adapter implementations run in-process and are
therefore trusted code. `SubprocessSandbox` only constrains child commands; it does not isolate
the main search process and is not filesystem or network isolation.

The local v0.1 registry can persist and restore only callback-free, content-addressed L0
`prompt_template` operators. Callbacks and adapters are not serialized; scheduler and memory-
policy persistence/evolution are future work. A production deployment needs an external
evaluator, a hardened container/VM for untrusted execution, and a run root signed and retained
by an external identity.

“Sealed” means excluded from foundry prompts and selection feedback during the protocol. It does
not make a task secret, and the sealed examples published in this repository are public.

## Primary metric

The v0.1 offline demo computes `improvement_acceleration` from two causal, sealed-search
trajectories. G0 runs `Improve@1.0.0` four times; G1 replaces only the first operation with the
installed evolved operator and then runs the same three Improve operations. Both trajectories
start from the same evaluated seed node and use the same task, seed, evaluator, model, sampling,
attempt ceiling, token ceiling, time ceiling, and cost ceiling.

For each of the four planned operations, the receipt records cumulative best utility and
cumulative resource usage. `best_utility_auc_per_attempt` is the arithmetic mean of cumulative
best utility after operations 1–4. The demo defines:

```text
improvement_acceleration = mean(AUC_per_attempt(G1)) - mean(AUC_per_attempt(G0))
```

The sealed JSON persists both full trajectories, operator content hashes, node/artifact lineage,
per-step evaluator usage, and per-arm model-generation call counts; replay reconstructs them from
the installed operator, canonical Improve runtime, and fixed task adapter. A baseline wrapper that
merely reuses the Improve passport or content hash is not accepted as the generation-zero control.
This bounded AUC delta measures earlier progress under a fixed operation horizon. It is not a
claim of unbounded acceleration, autonomous evaluator improvement, or model-weight recursion. A
project should say “stronger search” when final score rises but improvement rate does not.
