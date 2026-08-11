"""Fail-fast publication audit for the EvolSA source tree."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolsa.operators import seed_operators  # noqa: E402
from evolsa.schemas import (  # noqa: E402
    operator_passport_document,
    operator_passport_from_document,
)
REQUIRED = (
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "GOVERNANCE.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "pyproject.toml",
    "MANIFEST.in",
    "setup.py",
    "scripts/check_release.py",
    "src/evolsa/__init__.py",
    "src/evolsa/__main__.py",
    "src/evolsa/py.typed",
    "src/evolsa/schemas/__init__.py",
    "src/evolsa/schemas/operator-passport.schema.json",
    "schemas/operator-passport.schema.json",
    "docs/architecture.md",
    "docs/threat-model.md",
    "docs/releasing.md",
    "benchmarks/evolsa_bench_lite/manifest.json",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/security.yml",
    "operators/core/draft.json",
    "operators/core/improve.json",
    "operators/core/debug.json",
    "operators/core/crossover.json",
)
BAD_MARKERS = ("TODO", "TBD", "CHANGEME", "your-email@example.com", "example.com/repo")
ROOT_SCHEMA = "schemas/operator-passport.schema.json"
PACKAGE_SCHEMA = "src/evolsa/schemas/operator-passport.schema.json"


def local_links(markdown: str) -> list[str]:
    markdown_targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown)
    html_targets = re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", markdown)
    return [
        target.split("#", 1)[0]
        for target in (*markdown_targets, *html_targets)
        if target and not target.startswith(("http://", "https://", "mailto:", "#"))
    ]


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_system = metadata.get("build-system", {})
    if build_system.get("build-backend") != "setuptools.build_meta":
        errors.append("build backend must be setuptools.build_meta")
    if build_system.get("requires") != ["setuptools==80.9.0"]:
        errors.append("build-system must exactly pin the audited setuptools backend")

    package_data = metadata.get("tool", {}).get("setuptools", {}).get("package-data", {})
    schema_globs = package_data.get("evolsa.schemas", [])
    if "*.json" not in schema_globs:
        errors.append("pyproject.toml does not include evolsa.schemas JSON package data")
    if "py.typed" not in package_data.get("evolsa", []):
        errors.append("pyproject.toml does not ship the py.typed marker")

    scripts = metadata.get("project", {}).get("scripts", {})
    if scripts.get("evolsa") != "evolsa.cli:main":
        errors.append("console entry point must be evolsa.cli:main")
    if metadata.get("project", {}).get("license") != "Apache-2.0":
        errors.append("project license must be Apache-2.0")
    project_urls = metadata.get("project", {}).get("urls", {})
    if project_urls.get("Repository") != "https://github.com/sunyifei-126/EvolSA":
        errors.append("project Repository URL must identify the canonical GitHub repository")

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for directive in (
        "include SECURITY.md",
        "include GOVERNANCE.md",
        "include CONTRIBUTING.md",
        "include CODE_OF_CONDUCT.md",
        "include Makefile",
        "recursive-include scripts *.py",
        "recursive-include .github *.yml *.md",
        "recursive-include schemas *.json",
        "recursive-include src/evolsa/schemas *.json",
    ):
        if directive not in manifest:
            errors.append(f"MANIFEST.in is missing: {directive}")

    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    for workflow_path in workflow_paths:
        workflow_text = workflow_path.read_text(encoding="utf-8")
        if "ubuntu-latest" in workflow_text:
            errors.append(f"{workflow_path.relative_to(ROOT)} uses a floating runner image")
        for line in workflow_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- uses:"):
                continue
            action = stripped.removeprefix("- uses:").split("#", 1)[0].strip()
            reference = action.rsplit("@", 1)[-1]
            if not re.fullmatch(r"[0-9a-f]{40}", reference):
                errors.append(
                    f"{workflow_path.relative_to(ROOT)} has an unpinned action: {action}"
                )
    release_workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    for release_control in (
        "contents: read",
        "build-and-verify:",
        "publish:",
        "contents: write",
        "persist-credentials: false",
        "environment: release",
    ):
        if release_control not in release_workflow:
            errors.append(f"release workflow is missing control: {release_control}")

    version = metadata["project"]["version"]
    init_text = (ROOT / "src/evolsa/__init__.py").read_text(encoding="utf-8")
    if f'__version__ = "{version}"' not in init_text:
        errors.append("package version does not match pyproject.toml")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if f"version: {version}" not in citation:
        errors.append("CITATION.cff version does not match")
    if 'repository-code: "https://github.com/sunyifei-126/EvolSA"' not in citation:
        errors.append("CITATION.cff does not identify the canonical repository")

    root_schema_bytes = (ROOT / ROOT_SCHEMA).read_bytes()
    package_schema_bytes = (ROOT / PACKAGE_SCHEMA).read_bytes()
    if root_schema_bytes != package_schema_bytes:
        errors.append(
            "repository and packaged operator passport schemas differ: "
            f"{sha256(root_schema_bytes).hexdigest()} != {sha256(package_schema_bytes).hexdigest()}"
        )
    schema = json.loads(root_schema_bytes)
    if schema.get("properties", {}).get("api_version", {}).get("const") != "evolsa/v1":
        errors.append("operator passport schema api_version is not evolsa/v1")
    expected_required = {
        "api_version", "name", "version", "intent", "inputs", "preconditions",
        "expected_effects", "forbidden_effects", "context_policy", "generator",
        "budget", "verification", "lifecycle",
    }
    if set(schema.get("required", [])) != expected_required:
        errors.append("operator passport schema has an unexpected required-field contract")
    runtime_passports = {
        operator.passport.name: operator.passport for operator in seed_operators()
    }
    passport_paths = sorted((ROOT / "operators/core").glob("*.json"))
    if {path.stem for path in passport_paths} != set(runtime_passports):
        errors.append("static seed passport names do not match runtime seed operators")
    for passport_path in passport_paths:
        passport = json.loads(passport_path.read_text(encoding="utf-8"))
        missing = set(schema["required"]) - set(passport)
        if missing:
            errors.append(f"{passport_path.relative_to(ROOT)} missing {sorted(missing)}")
        if passport.get("api_version") != "evolsa/v1":
            errors.append(f"{passport_path.relative_to(ROOT)} has wrong api_version")
        protected = {"evaluator", "sandbox", "budget", "promotion_gate", "event_log", "release_permission"}
        if not protected.issubset(set(passport.get("forbidden_effects", []))):
            errors.append(f"{passport_path.relative_to(ROOT)} does not forbid every protected surface")
        try:
            parsed = operator_passport_from_document(passport)
        except (TypeError, ValueError) as error:
            errors.append(
                f"{passport_path.relative_to(ROOT)} fails strict passport parsing: {error}"
            )
            continue
        runtime = runtime_passports.get(passport_path.stem)
        if runtime is None or parsed != runtime:
            errors.append(
                f"{passport_path.relative_to(ROOT)} differs from its runtime seed passport"
            )
        elif passport != operator_passport_document(runtime):
            errors.append(
                f"{passport_path.relative_to(ROOT)} is not the canonical runtime passport document"
            )

    checked_links = 0
    for readme_name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / readme_name).read_text(encoding="utf-8")
        for marker in BAD_MARKERS:
            if marker in text:
                errors.append(f"{readme_name} contains publication marker {marker!r}")

    markdown_paths = sorted(
        path
        for path in ROOT.rglob("*.md")
        if not {"build", "dist", "__pycache__"} & set(path.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
    )
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for target in local_links(text):
            checked_links += 1
            candidate = (
                ROOT / target.lstrip("/")
                if target.startswith("/")
                else markdown_path.parent / target
            ).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(
                    f"local link escapes the release tree in {markdown_path.relative_to(ROOT)}: {target}"
                )
                continue
            if not candidate.exists():
                errors.append(
                    f"broken local link in {markdown_path.relative_to(ROOT)}: {target}"
                )

    forbidden = [path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_file()
                 and (path.suffix.lower() == ".pdf" or "reference" in path.name.lower())]
    if forbidden:
        errors.append(f"publication tree contains reference/PDF files: {forbidden}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"release audit passed: version {version}, {len(REQUIRED)} required files, "
        f"{len(list((ROOT / 'operators/core').glob('*.json')))} seed passports, "
        f"{checked_links} local links, packaged schema sha256={sha256(root_schema_bytes).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
