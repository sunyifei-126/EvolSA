# Evidence contract

Every promotion receipt must be reproducible from fixed, content-addressed inputs:

- candidate and baseline passport hashes;
- task-pack, split, evaluator, model, parent, seed, and budget identities;
- raw candidate/baseline outcomes for every paired trial;
- interval method and random seed;
- validity, cost, transfer, retention, regression, and integrity evidence;
- promotion policy hash and exact gate reasons;
- registry generation before and after installation.

Aggregate metrics are conveniences, not authority. The gate must recompute them from raw paired
outcomes or accept an attestation from a trusted aggregator bound to those outcomes. Candidate-
supplied metrics, directions, baselines, or boolean attestations cannot satisfy a hard gate.
Only pairs in which both candidate and baseline passed evaluator validity may contribute to
paired lift or its confidence interval. All attempts remain in the report for validity-rate,
cost, and provenance accounting; invalid scores can never raise promotion evidence.

The local event chain is an integrity mechanism: verification can detect later mutation or a
broken chain. It is not authentication, does not prove that the producer was honest, and cannot
stop an attacker controlling the host from replacing both the log and its local head. Production
deployments must verify the evidence, sign the resulting run root with an external identity, and
retain that signed root in an external storage domain.

The v0.1 evaluator adapter and gate run as trusted in-process code. Production evidence requires
an evaluator outside the agent/search process and untrusted execution inside a hardened
container or VM; a local hash chain alone cannot establish either property.
