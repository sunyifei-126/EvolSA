"""Self-contained evidence and operator-lineage reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evolsa.core import ExperienceCard, GateResult, LabReport, OperatorPassport
from evolsa.core.serde import pretty_json, to_primitive


_STYLE = """
:root{color-scheme:dark;--bg:#07111f;--panel:#0e1c2f;--line:#243d5a;--text:#e8f1ff;
--muted:#98aac1;--cyan:#42d7e8;--green:#58df93;--yellow:#ffd166;--red:#ff6b79}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#12314a 0,
var(--bg) 38%);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:1180px;margin:auto;padding:44px 24px 72px}h1{font-size:clamp(30px,5vw,58px);
line-height:1.05;margin:0 0 12px}h2{margin-top:42px}.kicker{color:var(--cyan);font-weight:800;
letter-spacing:.15em;text-transform:uppercase}.lede{color:var(--muted);max-width:780px;font-size:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card{background:
linear-gradient(145deg,rgba(20,42,66,.95),rgba(10,24,42,.95));border:1px solid var(--line);
border-radius:15px;padding:18px;box-shadow:0 12px 36px rgba(0,0,0,.22)}.metric{font-size:29px;
font-weight:800;color:var(--cyan)}.muted{color:var(--muted)}.pill{display:inline-block;padding:3px 9px;
margin:2px;border:1px solid var(--line);border-radius:999px;font-size:12px}.stable,.promote{color:var(--green)}
.shadow,.candidate,.revise{color:var(--yellow)}.rejected,.reject,.quarantine,.quarantined{color:var(--red)}
.node{position:relative}.node:before{content:"";position:absolute;left:-8px;top:20px;width:4px;height:36px;
border-radius:3px;background:var(--cyan)}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{white-space:pre-wrap;word-break:break-word;background:#07101d;padding:13px;border-radius:10px;
border:1px solid #203550;max-height:360px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--cyan)}a{color:var(--cyan)}
.graph{overflow:auto;padding:10px}.graph svg{display:block;min-width:760px;width:100%;height:auto}.edge{fill:none;
stroke:#4e7599;stroke-width:2}.graph-node rect{fill:#102842;stroke:#42d7e8;stroke-width:1.5;rx:10}
.graph-node text{fill:#e8f1ff;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.graph-node .life{fill:#58df93;font-size:10px}
"""


def _class(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum() or character in "-_")


def _render_operator_graph(passports: Sequence[OperatorPassport]) -> str:
    """Render a deterministic, script-free parent graph as inline SVG."""

    if not passports:
        return '<p class="muted">No operator passports.</p>'
    by_id = {passport.operator_id: passport for passport in passports}
    levels: dict[str, int] = {}

    def level(operator_id: str, trail: frozenset[str] = frozenset()) -> int:
        if operator_id in levels:
            return levels[operator_id]
        if operator_id in trail:
            return 0
        known_parents = tuple(
            parent for parent in by_id[operator_id].parents if parent in by_id
        )
        value = 0 if not known_parents else 1 + max(
            level(parent, trail | {operator_id}) for parent in known_parents
        )
        levels[operator_id] = value
        return value

    for operator_id in by_id:
        level(operator_id)
    grouped: dict[int, list[str]] = {}
    for operator_id in sorted(by_id):
        grouped.setdefault(levels[operator_id], []).append(operator_id)
    width = 1000
    node_width = 210
    node_height = 62
    vertical_gap = 125
    positions: dict[str, tuple[float, float]] = {}
    for row, operator_ids in sorted(grouped.items()):
        slot = width / (len(operator_ids) + 1)
        for column, operator_id in enumerate(operator_ids, start=1):
            positions[operator_id] = (slot * column - node_width / 2, 32 + row * vertical_gap)
    height = max(y for _, y in positions.values()) + node_height + 34

    edges: list[str] = []
    for child_id, passport in sorted(by_id.items()):
        child_x, child_y = positions[child_id]
        for parent_id in sorted(parent for parent in passport.parents if parent in positions):
            parent_x, parent_y = positions[parent_id]
            x1 = parent_x + node_width / 2
            y1 = parent_y + node_height
            x2 = child_x + node_width / 2
            y2 = child_y
            middle = (y1 + y2) / 2
            edges.append(
                f'<path class="edge" marker-end="url(#arrow)" '
                f'd="M{x1:.1f},{y1:.1f} C{x1:.1f},{middle:.1f} {x2:.1f},{middle:.1f} {x2:.1f},{y2:.1f}"/>'
            )

    nodes: list[str] = []
    for operator_id, passport in sorted(by_id.items()):
        x, y = positions[operator_id]
        label = operator_id if len(operator_id) <= 29 else operator_id[:26] + "…"
        nodes.append(
            f'<g class="graph-node" transform="translate({x:.1f},{y:.1f})">'
            f'<rect width="{node_width}" height="{node_height}"/>'
            f'<text x="12" y="25">{html.escape(label)}</text>'
            f'<text class="life" x="12" y="46">{html.escape(passport.lifecycle.value)}</text>'
            "</g>"
        )
    return (
        f'<div class="card graph" role="img" aria-label="Operator parent lineage graph">'
        f'<svg viewBox="0 0 {width} {height:.0f}" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        'fill="#4e7599"/></marker></defs>'
        f'{"".join(edges)}{"".join(nodes)}</svg></div>'
    )


def render_lineage_html(
    *,
    run_id: str,
    cards: Sequence[ExperienceCard],
    passports: Sequence[OperatorPassport],
    reports: Sequence[LabReport] = (),
    gates: Sequence[GateResult] = (),
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Render a standalone audit ledger without scripts or external assets."""

    metadata = metadata or {}
    passport_by_id = {passport.operator_id: passport for passport in passports}
    report_by_operator = {report.operator_id: report for report in reports}
    gate_by_operator = {gate.operator_id: gate for gate in gates}
    best = max((card.score for card in cards), default=0.0)
    new_best = sum(1 for card in cards if card.current_best)
    operators_used = sorted({card.operator_id for card in cards})
    operator_graph = _render_operator_graph(passports)

    operator_cards: list[str] = []
    for passport in passports:
        report = report_by_operator.get(passport.operator_id)
        gate = gate_by_operator.get(passport.operator_id)
        parents = ", ".join(passport.parents) or "genesis"
        evidence = ""
        if report:
            evidence = (
                f'<p><span class="metric">{report.mean_lift:+.3f}</span><br>'
                f'<span class="muted">paired lift · 95% CI [{report.lcb95:+.3f}, {report.ucb95:+.3f}]</span></p>'
            )
        if gate:
            evidence += f'<p class="{_class(gate.decision.value)}"><strong>{html.escape(gate.decision.value.upper())}</strong> · {html.escape("; ".join(gate.reasons))}</p>'
        operator_cards.append(
            '<article class="card node">'
            f'<div class="kicker">{html.escape(passport.lifecycle.value)}</div>'
            f'<h3>{html.escape(passport.operator_id)}</h3>'
            f'<p>{html.escape(passport.intent)}</p><p class="muted">parents: {html.escape(parents)}</p>'
            f'{evidence}<details><summary>Passport</summary><pre>{html.escape(pretty_json(passport))}</pre></details>'
            '</article>'
        )

    rows: list[str] = []
    for card in cards:
        passport = passport_by_id.get(card.operator_id)
        label = passport.name if passport else card.operator_id
        rows.append(
            "<tr>"
            f"<td>{card.step_id}</td><td>{html.escape(card.node_id)}</td><td>{html.escape(label)}</td>"
            f"<td>{html.escape(', '.join(card.parent_node_ids) or 'root')}</td>"
            f"<td>{card.score:.4f}</td><td>{card.delta_vs_parent:+.4f}</td>"
            f"<td>{card.novelty_score:.3f}</td><td>{html.escape(card.status.value)}</td>"
            f"<td>{card.runtime_seconds:.3f}s / {card.token_count} tok</td>"
            "</tr>"
        )

    safe_payload = json.dumps(
        {
            "run_id": run_id,
            "cards": [to_primitive(card) for card in cards],
            "passports": [to_primitive(passport) for passport in passports],
            "reports": [to_primitive(report) for report in reports],
            "gates": [to_primitive(gate) for gate in gates],
            "metadata": to_primitive(metadata),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvolSA · {html.escape(run_id)}</title><style>{_STYLE}</style></head><body><main>
<div class="kicker">Bounded Recursive Self-Improvement</div><h1>Evolve the improver,<br>not just the answer.</h1>
<p class="lede">Run <code>{html.escape(run_id)}</code> exposes every operator, parent, execution outcome,
matched-budget experiment, and promotion decision. The evaluator and promotion kernel remain outside the evolvable surface.</p>
<section class="grid"><div class="card"><div class="metric">{len(cards)}</div><div class="muted">evaluated nodes</div></div>
<div class="card"><div class="metric">{best:.4f}</div><div class="muted">best observed score</div></div>
<div class="card"><div class="metric">{new_best}</div><div class="muted">incumbent updates</div></div>
<div class="card"><div class="metric">{len(operators_used)}</div><div class="muted">operators exercised</div></div></section>
<h2>Operator lineage</h2>{operator_graph}
<h2>Operator passports</h2><section class="grid">{''.join(operator_cards) or '<p class="muted">No operator passports.</p>'}</section>
<h2>Execution evidence</h2><div class="card"><table><thead><tr><th>Step</th><th>Node</th><th>Operator</th>
<th>Parents</th><th>Score</th><th>Δ parent</th><th>Novelty</th><th>Status</th><th>Budget</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>Run metadata</h2><pre>{html.escape(pretty_json(metadata))}</pre>
<details><summary>Machine-readable evidence bundle</summary><pre>{html.escape(safe_payload)}</pre></details>
</main></body></html>"""


def render_review_markdown(
    *,
    run_id: str,
    cards: Iterable[ExperienceCard],
    reports: Iterable[LabReport],
    gates: Iterable[GateResult],
) -> str:
    lines = [f"# EvolSA review: `{run_id}`", "", "## Execution lineage", ""]
    for card in cards:
        lines.append(
            f"- step {card.step_id}: `{card.node_id}` via `{card.operator_id}` — "
            f"score {card.score:.4f}, delta {card.delta_vs_parent:+.4f}, status `{card.status.value}`"
        )
    lines.extend(["", "## Matched-budget operator lab", ""])
    for report in reports:
        lines.append(
            f"- `{report.operator_id}` vs `{report.baseline_id}`: mean lift {report.mean_lift:+.4f}, "
            f"95% CI [{report.lcb95:+.4f}, {report.ucb95:+.4f}], coverage {report.transfer_coverage}"
        )
    lines.extend(["", "## Promotion decisions", ""])
    for gate in gates:
        lines.append(f"- **{gate.decision.value.upper()}** `{gate.operator_id}`: {'; '.join(gate.reasons)}")
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)
    return destination
