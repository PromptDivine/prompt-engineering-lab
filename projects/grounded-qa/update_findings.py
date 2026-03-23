"""
update_findings.py
==================
Grounded QA — Auto-populate findings
Project: P5 · prompt-engineering-lab

Run AFTER run_experiment.py to auto-fill:
  - README.md results tables
  - Notebook key findings (printed — paste in)

Usage:
    python update_findings.py
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")


def load():
    df  = pd.read_csv(RESULTS_DIR / "results.csv")
    lb  = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    hr  = pd.read_csv(RESULTS_DIR / "hallucination_report.csv")
    df  = df[df["error"].isna() | (df["error"] == "")].copy()
    return df, lb, hr


def build_leaderboard_table(lb: pd.DataFrame) -> str:
    overall = lb[lb["condition"] == "OVERALL"].sort_values("factual_accuracy", ascending=False)
    rows = [
        "| Rank | Model | Factual Accuracy | Grounding Rate | Hallucination Rate | Latency |",
        "|------|-------|-----------------|----------------|-------------------|---------|",
    ]
    for rank, (_, row) in enumerate(overall.iterrows(), 1):
        hall = row.get("hallucination_rate", row.get("hallucination_flag", 0))
        rows.append(
            f"| {rank} | {row['model']} "
            f"| {row.get('factual_accuracy',0):.1%} "
            f"| {row.get('grounding_rate',0):.1%} "
            f"| {hall:.1%} "
            f"| {row.get('latency_s',0):.2f}s |"
        )
    return "\n".join(rows)


def build_condition_table(lb: pd.DataFrame) -> str:
    cats = lb[lb["condition"] != "OVERALL"]
    rows = [
        "| Condition | Avg Factual Accuracy | Avg Hallucination Rate |",
        "|-----------|---------------------|------------------------|",
    ]
    for cond in ["ungrounded", "grounded", "cited"]:
        sub = cats[cats["condition"] == cond]
        if sub.empty:
            continue
        acc  = sub["factual_accuracy"].mean() if "factual_accuracy" in sub.columns else 0
        hall = sub.get("hallucination_rate", sub.get("hallucination_flag", pd.Series([0]))).mean()
        rows.append(f"| {cond} | {acc:.1%} | {hall:.1%} |")
    return "\n".join(rows)


def update_readme(lb_table: str, cond_table: str):
    content = README_PATH.read_text(encoding="utf-8")

    markers = [
        ("| Rank | Model | Factual Accuracy |", "\n\n*Run", lb_table),
        ("| Condition | Avg Factual Accuracy |", "\n\n---", cond_table),
    ]
    for start_marker, end_marker, replacement in markers:
        si = content.find(start_marker)
        ei = content.find(end_marker, si)
        if si != -1 and ei != -1:
            content = content[:si] + replacement + content[ei:]

    README_PATH.write_text(content, encoding="utf-8")
    print("  README.md updated.")


def print_notebook_findings(df: pd.DataFrame, lb: pd.DataFrame, hr: pd.DataFrame):
    overall = lb[lb["condition"] == "OVERALL"].sort_values("factual_accuracy", ascending=False)
    best_model  = overall.iloc[0]["model"] if not overall.empty else "N/A"
    best_acc    = overall.iloc[0].get("factual_accuracy", 0) if not overall.empty else 0
    worst_model = overall.iloc[-1]["model"] if len(overall) > 1 else "N/A"

    # Hallucination reduction from ungrounded → grounded
    ung = df[df["condition"]=="ungrounded"]["hallucination_flag"].mean() if "ungrounded" in df["condition"].values else 0
    grd = df[df["condition"]=="grounded"]["hallucination_flag"].mean()   if "grounded"   in df["condition"].values else 0
    reduction = ung - grd

    # Unanswerable handling
    unanswerable = df[df["is_answerable"] == False]
    refusal_rate = unanswerable["unanswerable_correct"].mean() if not unanswerable.empty else 0

    # Top hallucination type
    if not hr.empty:
        top_type = hr.groupby("hallucination_type")["count"].sum().sort_values(ascending=False)
        top_hall_type = top_type.index[0] if not top_type.empty else "N/A"
        top_hall_pct  = top_type.iloc[0] / top_type.sum() * 100 if not top_type.empty else 0
    else:
        top_hall_type, top_hall_pct = "N/A", 0

    # Citation validity
    cited = df[df["condition"]=="cited"]
    cit_valid = cited["citation_valid"].mean() if not cited.empty and "citation_valid" in cited.columns else 0

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Best model overall:** `{best_model}` — {best_acc:.1%} factual accuracy
2. **Grounding impact:** Hallucination rate dropped {reduction:.1%} (ungrounded={ung:.1%} → grounded={grd:.1%})
3. **Unanswerable handling:** Models correctly refused {refusal_rate:.1%} of unanswerable questions
4. **Most common hallucination:** `{top_hall_type}` — {top_hall_pct:.0f}% of all hallucinations
5. **Citation validity:** {cit_valid:.1%} of model-generated citations verified against context
6. **Key insight:** [Fill in after reviewing chart_grounding_improvement.png]

---
*See `results/hallucination_report.csv` for per-model breakdown.*
*See `results/rag_results.csv` for RAG pipeline retrieval accuracy.*
""")


def main():
    print("Loading results...")
    df, lb, hr = load()
    print(f"  {len(df)} rows | {df['model'].nunique()} models\n")

    print("Building README tables...")
    update_readme(build_leaderboard_table(lb), build_condition_table(lb))
    print_notebook_findings(df, lb, hr)
    print("\nDone.")


if __name__ == "__main__":
    main()
