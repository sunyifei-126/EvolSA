# Research foundations

EvolSA is informed by *Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement
in Machine Learning Engineering* (Yang et al., 2026, arXiv:2607.28568). The paper contributes
three connected layers: executable task environments, execution-grounded learning of Draft /
Improve / Debug / Crossover, and long-horizon search using deterministic experience records,
quality-progress-novelty parent selection, and operation-triggered memory.

The implementation adopts four engineering principles:

- training and inference should share an operator-shaped action space;
- evaluator-owned outcomes, not model self-ratings, define quality; the v0.1 trusted adapter is
  an interface demonstration, while production requires an evaluator outside the agent process;
- memory is synthesized after selecting an operation and is branch-local and bounded;
- search allocates budget to quality, parent-relative progress, and underexplored method families.

EvolSA investigates a limitation identified explicitly in the paper: candidate solutions evolve,
while the evolutionary system itself remains largely fixed. The design treats the operator set,
scheduler, and memory policy as a bounded conceptual genome subject to matched-budget held-out
evaluation. The v0.1 executable scope is narrower: only callback-free L0 prompt-template
operators can be persisted and restored; scheduler and memory-policy evolution remain roadmap
work. Its trace-tag-driven synthesis is bounded pattern instantiation, not a claim of autonomous
discovery of arbitrary or human-unenumerated strategies.

This repository is a from-scratch implementation. It does not redistribute the supplied paper,
OpenMLE datasets, model weights, or OpenRSI source. Optional adapters must respect upstream
licenses and preserve attribution.
