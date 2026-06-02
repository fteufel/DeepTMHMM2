import json
import os
import sys
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
try:
    with open("results/predictions.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("ERROR: results/predictions.json not found.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_prediction(entry: dict) -> bool:
    """Return True if this entry is a real prediction (has an 'id' field)."""
    return "id" in entry


def render_entry(entry: dict) -> str:
    """Render a single prediction entry as a Markdown block."""
    lines = []

    seq_id   = entry["id"]
    prot_type = entry["type"]

    # --- Header ---
    lines.append(f"## {seq_id} | {prot_type}")
    lines.append("")

    # --- Membrane types (only for non-globular) ---
    if not prot_type.lower().startswith("globular"):
        mem_types = entry.get("membrane_types", [])
        if mem_types:
            lines.append(f"**Membrane type(s):** {', '.join(mem_types)}")
        else:
            lines.append("**Membrane type(s):** —")
        lines.append("")

    # --- Embedded figure + per-entry download ---
    figure_path = f"{seq_id}_plot.png"       # relative path for Markdown
    figure_abs  = os.path.join("results", figure_path)
    if os.path.exists(figure_abs):
        lines.append(f"**Download:** [PNG]({figure_path})")
        lines.append("")
        lines.append(f"![Topology plot for {seq_id}]({figure_path})")
        lines.append("")

    # --- Segments table ---
    segments = entry.get("segments", [])
    if segments:
        rows = [[seg[0], seg[1], seg[2], seg[2] - seg[1] + 1] for seg in segments]
        table = tabulate(
            rows,
            headers=["Region", "Start", "End", "Length"],
            tablefmt="github",
        )
        lines.append("**Segments:**")
        lines.append("")
        lines.append(table)
    else:
        lines.append("**Segments:** —")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------
output_md = "# DeepTMHMM 2 — Predictions\n\n"

output_md += "### Downloads\n\n"
output_md += "[JSON summary](predictions.json) / "
output_md += "[Topology strings](predicted_topologies.3line) / "
output_md += "[GFF3](TMRs.gff3) / "
output_md += "[Membrane type probabilities](membrane_types.tsv)\n\n"
output_md += "---\n\n"

predictions = [e for e in results if is_prediction(e)]

if not predictions:
    output_md += "No predictions available.\n"
else:
    for entry in predictions:
        output_md += render_entry(entry)

with open("output.md", "w") as f:
    f.write(output_md)

print(f"Report written to output.md ({len(predictions)} prediction(s)).")
