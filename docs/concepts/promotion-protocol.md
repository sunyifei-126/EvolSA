# Matched-Budget Promotion Protocol

The Operator Lab compares a candidate with its declared baseline using identical task IDs,
splits, seeds, parents, model configuration, sampling parameters, token limits, execution limits,
attempt counts, and evaluator version.

The host-owned `PromotionPolicy` pins the baseline operator ID and immutable content hash, exact
`(task_id, seed)` allocation, required split and evaluator ID, metric direction, bootstrap sample
count/seed/confidence, and cost penalty. A report cannot select weaker statistics, relabel a
mining split, flip metric direction, or substitute an unregistered baseline and still promote.

For paired outcomes `d_i = score(candidate_i) - score(baseline_i)`, EvolSA reports mean lift and
a deterministic bootstrap 95% interval over jointly valid pairs only. Invalid attempts remain
visible in validity-rate and cost accounting, but their scores cannot increase lift. Promotion
is fail-closed:

```text
LCB95(paired lift) > minimum quality delta
valid_rate(candidate) >= valid_rate(baseline) - tolerance
cost-adjusted lift > minimum efficiency delta
worst regression <= maximum allowed regression
transfer coverage >= minimum tasks
matched budget = true
integrity and safety checks = pass
```

Missing or contradictory evidence cannot promote. An operational performance failure rejects;
an integrity or protected-surface failure quarantines. The gate stores the policy hash with the
decision so later configuration changes cannot silently reinterpret an old result.

Mine, validation, promotion, and sealed-test splits must be disjoint. Sealed results are for
cross-generation reporting and must not drive foundry revisions. Here “sealed” describes this
information-flow rule; it does not imply secrecy. Task packs included in the public repository
are public. Production secrecy and independent evaluation require the sealed split and evaluator
to live outside the agent/search process.

The v0.1 gate and evaluator adapter are trusted in-process components. For production, execute
the evaluator externally, run untrusted candidate code in a hardened container/VM, and bind the
verified receipt to an externally signed and retained run root.
