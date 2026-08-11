# Security policy

## Supported versions

EvolSA is a research preview. Security fixes target the latest release and `main`.

## Reporting

Do not open a public issue for a vulnerability. Use GitHub private vulnerability reporting when
available, or contact the repository maintainers privately. Include the affected version, threat
boundary, reproduction, impact, and suggested containment. Do not include live credentials,
private datasets, or exploit payloads against systems you do not own.

## Deployment warning

`SubprocessSandbox` is a deterministic development backend, not a security boundary. It cannot
enforce network denial or protect the host from hostile native code. Production deployments must
run untrusted artifacts in a hardened container or VM with a separate identity, read-only
evaluator mounts, network policy, resource quotas, output limits, and external audit storage.

EvolSA treats evaluator, sandbox policy, hard budget, event history, promotion policy, signature,
and release permissions as protected surfaces. A passport that declares or attempts a protected
effect must be quarantined before execution.
