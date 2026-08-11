# Quickstart

Install the editable package with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the offline bounded-RSI demo and inspect its evidence:

```bash
evolsa demo --output .evolsa/demo
evolsa inspect .evolsa/demo
```

The output directory contains a run manifest, raw experience cards, a synthesized passport,
paired lab evidence with an executable Improve baseline, a promotion receipt, registry state,
matched sealed G0/G1 search trajectories with per-step utility/usage receipts, Markdown review,
and `index.html`.

The demo is an integrity-checkable local experiment, not a secure deployment. Its event hash
chain detects inconsistent local history but does not authenticate the producer. Its “sealed”
split is excluded from foundry feedback, but the task cases shipped in this public repository are
not secret. The in-process callbacks/adapters are trusted code, and `SubprocessSandbox` only
constrains a child command—it does not isolate the main search process or deny filesystem/network
access. Only callback-free L0 prompt-template operators can be restored from the local registry.

To connect another domain, implement `TaskAdapter.prepare`, `seed_artifact`, `execute`, and
`score`. An in-process adapter must be trusted. Never include evaluator-only state in
`TaskState.metadata` passed to an operator; keeping it merely inside a local adapter does not make
it secret from a compromised host. Production use requires an external evaluator, hardened
container/VM execution for untrusted artifacts, and externally signed and retained run roots.
