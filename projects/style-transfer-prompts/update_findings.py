"""
update_findings.py
==================
Style Transfer Prompts — Auto-populate findings after experiment run
Project: P2 · prompt-engineering-lab

Run this AFTER run_experiment.py to auto-fill:
  - README.md formality shift table
  - Key findings summary (printed to console — paste into notebook Cell 8)

Usage:
    python update_findings.py
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")

STYLE_ORDER = ["legal","academic","medical","executive","journalism",
               "technical","storytelling","marketing","minimalist","casual"]


def load_results() -> pd.DataFrame:
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        raise FileNotFoundError("No results.csv found. Run run_experiment.py first.")
    df = pd.read_csv(path)
    return df[df["error"].isna() | (df["error"] == "")].copy()


def build_table(df: pd.DataFrame) -> str:
    """Build the markdown formality table sorted by formality descending."""
    cols = {
        "formality_score":    "formality_score",
        "fk_grade":           "fk_grade",
        "compression_ratio":  "compression_ratio",
    }
    # Best strategy per style = highest formality_score
    best = (
        df.groupby(["style","strategy"])["formality_score"]
        .mean()
        .reset_index()
    )
    best_strat = best.loc[best.groupby("style")["formality_score"].idxmax()].set_index("style")["strategy"]

    agg = df.groupby("style")[list(cols.values())].mean()
    agg["winner_strategy"] = best_strat

    # Sort by formality descending
    agg = agg.sort_values("formality_score", ascending=False)

    rows = ["| Style | Avg Formality | FK Grade | Compression | Winner Strategy |",
            "|-------|--------------|----------|-------------|-----------------|"]

    for style, row in agg.iterrows():
        rows.append(
            f"| {style} "
            f"| {row['formality_score']:.3f} "
            f"| {row['fk_grade']:.1f} "
            f"| {row['compression_ratio']:.2f}x "
            f"| {row.get('winner_strategy', '—')} |"
        )

    return "\n".join(rows)


def update_readme(table_md: str):
    content = README_PATH.read_text(encoding="utf-8")

    # Find and replace the table block between the two header lines
    start_marker = "| Style | Avg Formality | FK Grade | Compression | Winner Strategy |"
    end_marker   = "*Run the experiment to populate."

    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("  Could not find table markers in README.md — printing table instead:\n")
        print(table_md)
        return

    updated = content[:start_idx] + table_md + "\n\n" + content[end_idx:]
    README_PATH.write_text(updated, encoding="utf-8")
    print("  README.md updated.")


def print_notebook_findings(df: pd.DataFrame):
    """Print the Cell 8 findings block — copy-paste into the notebook."""

    agg = df.groupby("style")[["formality_score","fk_grade","compression_ratio"]].mean()
    delta_cols = [c for c in ["delta_fk_grade","delta_formality","delta_sentiment"] if c in df.columns]

    # Best style for formality shift
    if "delta_formality" in df.columns:
        delta_form = df.groupby("style")["delta_formality"].mean()
        best_formal_style  = delta_form.idxmax()
        best_formal_delta  = delta_form.max()
        most_casual_style  = delta_form.idxmin()
        most_casual_delta  = delta_form.min()
    else:
        best_formal_style  = agg["formality_score"].idxmax()
        best_formal_delta  = agg["formality_score"].max()
        most_casual_style  = agg["formality_score"].idxmin()
        most_casual_delta  = agg["formality_score"].min()

    # Most readable
    most_readable = agg["fk_grade"].idxmin()
    most_readable_grade = agg["fk_grade"].min()

    # Most complex
    most_complex = agg["fk_grade"].idxmax()
    most_complex_grade = agg["fk_grade"].max()

    # Compression
    most_compressed = agg["compression_ratio"].idxmin()
    most_compressed_ratio = agg["compression_ratio"].min()
    most_expanded = agg["compression_ratio"].idxmax()
    most_expanded_ratio = agg["compression_ratio"].max()

    # Best strategy
    strat_agg = df.groupby("strategy")["formality_score"].mean()
    best_strategy = strat_agg.idxmax()
    best_strategy_score = strat_agg.max()

    # Best model (if judge scores available)
    if "judge_overall" in df.columns and df["judge_overall"].sum() > 0:
        model_judge = df.groupby("model")["judge_overall"].mean()
        best_model = model_judge.idxmax()
        best_model_score = model_judge.max()
        model_line = f"5. **Best model for style transfer:** {best_model} — avg judge score {best_model_score:.2f}/5"
    else:
        model_line = "5. **Best model for style transfer:** Run with --llm-judge to score style adherence per model"

    findings = f"""
## 8. Key Findings

1. **Highest formality shift:** `{best_formal_style}` — delta_formality = {best_formal_delta:+.3f}
2. **Most casual output:** `{most_casual_style}` — delta_formality = {most_casual_delta:+.3f}
3. **Most readable style:** `{most_readable}` — FK grade {most_readable_grade:.1f}
4. **Most complex style:** `{most_complex}` — FK grade {most_complex_grade:.1f}
5. **Most compressed output:** `{most_compressed}` — {most_compressed_ratio:.2f}x source length
6. **Most expanded output:** `{most_expanded}` — {most_expanded_ratio:.2f}x source length
7. **Best prompt strategy:** `{best_strategy}` — avg formality {best_strategy_score:.3f}
8. {model_line}

---
*See `results/gallery.html` for full side-by-side comparison of all outputs.*
"""
    print("\n" + "="*60)
    print("COPY THIS INTO NOTEBOOK CELL 8 (replace the placeholder):")
    print("="*60)
    print(findings)


def main():
    print("Loading results...")
    df = load_results()
    print(f"  {len(df)} valid rows | {df['model'].nunique()} models | {df['style'].nunique()} styles\n")

    print("Building README table...")
    table = build_table(df)
    update_readme(table)

    print_notebook_findings(df)
    print("\nDone. Commit your updated README.md.")


if __name__ == "__main__":
    main()
