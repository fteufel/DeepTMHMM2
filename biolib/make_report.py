import json
import pandas as pd
import os
import glob
from tabulate import tabulate

# Load predictions
try:
    with open("results/predictions.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

output_md = "# DeepTMHMM2 - Results\n\n"

if not results:
    output_md += "No predictions available.\n"
else:
    output_md += "## Summary\n\n"
    if isinstance(results, list):
        df = pd.DataFrame(results)
        # Select key columns for display
        cols = ['id', 'type', 'membrane_type', 'topology']
        available_cols = [c for c in cols if c in df.columns]
        output_md += tabulate(df[available_cols], tablefmt="github", headers="keys")
        output_md += "\n\n"
        
        # Display plots
        output_md += "## Visualizations\n\n"
        plot_files = glob.glob("results/*_plot.png")
        for plot in plot_files:
            plot_name = os.path.basename(plot)
            output_md += f"### {plot_name}\n"
            output_md += f"![{plot_name}]({plot_name})\n\n"
    else:
        output_md += str(results)

with open("output.md", "w") as f:
    f.write(output_md)
