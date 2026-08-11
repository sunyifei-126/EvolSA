# Threat model

## Trust boundary

Candidate artifacts and model-generated data are treated as untrusted. In v0.1, however, all
Python executed in the main process is inside the trusted computing base. This includes Python
operator callbacks, `TaskAdapter` and `ModelBackend` implementations, evaluator adapters, the
search controller, registry, and promotion gate. Loading an untrusted callback or adapter into
that process violates the v0.1 trust boundary.

`SubprocessSandbox` is process-level resource control for the child command it starts. It does
not isolate the main search process, prevent the child from reading paths available to the host
user, or deny child network access. Its name and workspace checks must not be interpreted as a
container, VM, filesystem, syscall, or network security boundary.

“Sealed” is an experimental information-flow designation: the split is withheld from foundry
prompts and selection feedback. A sealed task shipped in this public repository is public and
must not be treated as a secret. Confidential production tests need an external evaluator and a
separate access-controlled storage/execution domain.

## Primary threats

| Threat | Production control required (not implied by the v0.1 demo) |
|---|---|
| evaluator leakage or reward hacking | external evaluator, protected mounts, semantic hack checks, held-out splits |
| budget theft | kernel-owned accounting for tokens, attempts, wall time, cost, and external calls |
| high score with fragile behavior | multiple seeds, OOD/transfer tasks, worst regression and retention gates |
| malicious operator | passport signature, protected-effect validation, static scan, hardened network-denied container/VM, shadow lifecycle |
| event or aggregate forgery | content-addressed artifacts, hash-chain verification, recomputation from raw outcomes, externally signed run root |
| scheduler Goodharting | independent promotion tasks and policy hash; sealed tests never enter selection feedback |
| path/symlink escape | normalized relative paths, no symlink/gitlink artifacts, hardened container/VM filesystem boundary |
| unsafe installation | explicit stable transition, signed evidence bundle, retained prior generation, tested rollback |

## Non-goals

The Python reference kernel cannot defend against a hostile local user, compromised OS, kernel
exploit, an untrusted in-process callback/adapter, or a model with direct access to the evidence
database. The local event hash chain detects inconsistent history when verified, but it neither
authenticates the producer nor prevents an attacker controlling the host from replacing the log
and its local head together.

Do not execute untrusted code without a separately configured hardened container or VM. A
production deployment also needs an evaluator outside the agent/search process and an external
signing identity that signs and retains the verified run root. The repository does not provide
those production trust anchors.
