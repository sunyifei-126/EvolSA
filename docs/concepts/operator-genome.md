# Operator Genome and Passport

An Operator Passport is the auditable unit of EvolSA. It includes semantic version, intent,
lineage, inputs, applicability conditions, expected and forbidden effects, context policy,
generator implementation, hard budget, verification baseline, guardrails, evidence, lifecycle,
and known failures.

The bootstrap instruction set contains Draft, Improve, Debug, and Crossover. In v0.1 the foundry
can instantiate a more specialized L0 prompt/template operator when repeated evidence matches
the configured trace tags and synthesis rules. This bounded derivation should not be described
as autonomous discovery of arbitrary new skills or as proof that a human did not pre-enumerate
the relevant method family. Later releases may add L1 tool graphs, L2 Python operators, or
model-internal adapters without changing the passport or promotion contract.

Only callback-free, content-addressed L0 `prompt_template` operators can be persisted and
rehydrated by the v0.1 local registry. Python callbacks and adapter implementations execute as
trusted in-process code and are not serialized or restored. Supporting another generator type
requires a separately reviewed loader and, for untrusted code, an external isolation boundary.

## Lifecycle

```text
proposed → shadow → candidate → stable → deprecated
                  ↘ rejected / quarantined
```

Rejected passports remain in the negative-result registry. Quarantined passports cannot be
executed or used as parents until a human clears the integrity failure. Stable installation never
deletes the prior stable generation, so rollback is content-addressed.
