# Contributing

Thank you for helping build a reproducible bounded-RSI lab.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Use a focused branch and pull request. Add tests for observable behavior and update public docs
when changing contracts. Keep the core dependency-free unless a dependency is justified by a
measured capability or assurance improvement.

## Contribution paths

- TaskAdapter with deterministic evaluator and license-clean lite tasks;
- Operator Passport plus paired evidence and failure modes;
- promotion metric or statistical verifier;
- sandbox/attestation adapter;
- reproduction or negative result on another model/domain;
- documentation and threat-model improvements.

Do not commit model credentials, private datasets, generated run databases, supplied reference
PDFs, or code whose upstream license conflicts with Apache-2.0.
