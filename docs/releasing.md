# Releasing

Before tagging a release:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m evolsa demo --output /tmp/evolsa-release-demo
python -m evolsa replay /tmp/evolsa-release-demo
python -m build --no-isolation
```

Install the wheel into a fresh environment, run `evolsa --version`, and rerun the demo. Confirm
the version in `pyproject.toml`, `src/evolsa/__init__.py`, `CITATION.cff`, and `CHANGELOG.md`.
The checked-in release workflow installs the audited `setuptools==80.9.0` and `build==1.3.0`,
runs the source tests with the src-layout package installed, builds both distributions, then
installs the wheel into a clean environment before any write-capable release job starts.

Before the first public tag, protect `main`, add a `v*` tag ruleset, and configure the `release`
GitHub Environment with required maintainer approval. The workflow rejects a tagged commit that
is not an ancestor of `origin/main`. Its build job has read-only repository access and no persisted
checkout credential; only the final artifact-only job receives `contents: write`.

Review the repository tree for credentials, generated run databases, task answers, supplied
reference PDFs, model weights, and incompatible upstream code. A source release is not an
operator promotion: both require their own evidence and approval.

Do not describe a source release as production-safe merely because local replay and hash-chain
verification pass. The local chain checks integrity, not producer identity; public sealed tasks
are not secrets; in-process callbacks and adapters are trusted; and `SubprocessSandbox` does not
isolate the main search process. Also verify that release notes state that only callback-free L0
prompt-template operators can be persisted and restored.

Production deployment requires controls outside this source release: an evaluator outside the
agent/search process, a separately configured hardened container/VM for untrusted execution, and
a verified run root signed and retained by an external identity.
