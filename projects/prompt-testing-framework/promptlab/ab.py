"""
promptlab/ab.py
===============
A/B prompt comparison engine.
Compares two prompt variants across models and inputs.
Computes statistical significance via paired t-test and win rates.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ABReport:
    """Results of an A/B comparison between two prompts."""
    prompt_a_id: str
    prompt_b_id: str
    model: str
    n_inputs: int

    # Per-metric comparison
    metrics: dict = field(default_factory=dict)
    # { metric_name: {
    #     "mean_a": float, "mean_b": float,
    #     "delta": float, "delta_pct": float,
    #     "winner": "A"|"B"|"tie",
    #     "p_value": float, "significant": bool
    # }}

    # Overall verdict
    overall_winner: str = "tie"   # "A", "B", or "tie"
    recommendation: str = ""

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"A/B REPORT: {self.prompt_a_id} vs {self.prompt_b_id}",
            f"Model: {self.model} | N inputs: {self.n_inputs}",
            f"{'='*60}",
        ]
        for metric, data in self.metrics.items():
            sig = " *" if data.get("significant") else ""
            lines.append(
                f"  {metric:30s}  A={data['mean_a']:.3f}  B={data['mean_b']:.3f}"
                f"  Δ={data['delta']:+.3f} ({data['delta_pct']:+.1f}%)"
                f"  winner={data['winner']}{sig}"
            )
        lines.append(f"\n  OVERALL WINNER: {self.overall_winner}")
        lines.append(f"  {self.recommendation}")
        lines.append("  (* = statistically significant, p < 0.05)")
        return "\n".join(lines)


def _paired_ttest_p(a_scores: list, b_scores: list) -> float:
    """
    Simple paired t-test returning p-value.
    Uses pure Python — no scipy dependency.
    """
    n = len(a_scores)
    if n < 2:
        return 1.0

    diffs = [b - a for a, b in zip(a_scores, b_scores)]
    mean_d = sum(diffs) / n
    if mean_d == 0:
        return 1.0

    variance = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if variance == 0:
        return 0.0 if mean_d != 0 else 1.0

    std_err = math.sqrt(variance / n)
    t_stat = mean_d / std_err
    df = n - 1

    # Approximate p-value using t-distribution CDF approximation
    # (accurate enough for portfolio purposes; use scipy for production)
    x = df / (df + t_stat ** 2)
    # Beta function regularized approximation
    p_approx = _incomplete_beta(df / 2, 0.5, x)
    return min(1.0, max(0.0, p_approx))


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Rough approximation of regularized incomplete beta for p-value."""
    # Series expansion (works for moderate df values)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    try:
        import math
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        log_x = math.log(x)
        log_1mx = math.log(1 - x)
        s = 0.0
        term = 1.0
        for k in range(100):
            if k == 0:
                term = math.exp(a * log_x + b * log_1mx - lbeta) / a
            else:
                term *= (k - b) * x / (k * (a + k))
            s += term
            if abs(term) < 1e-8:
                break
        return min(1.0, max(0.0, s))
    except Exception:
        return 0.5  # fallback neutral p-value


class ABComparison:
    """
    Compare two prompt variants (A and B) on a set of inputs.

    Usage:
        from promptlab import PromptLabClient, ABComparison
        from promptlab.scorers import RubricScorer

        ab = ABComparison(
            client=PromptLabClient(),
            models=["gpt-4o-mini"],
            scorer=RubricScorer([...]),
        )

        report = ab.compare(
            prompt_a="Summarize this briefly: {{text}}",
            prompt_b="Give a 2-sentence summary: {{text}}",
            inputs=[{"id": "doc1", "text": "..."}, ...],
            prompt_a_id="brief",
            prompt_b_id="two_sentence",
        )
        print(report.summary())
    """

    def __init__(
        self,
        client=None,
        models: Optional[list] = None,
        scorer=None,
        primary_metric: str = "rubric_score",
    ):
        from .client import PromptLabClient
        self.client         = client or PromptLabClient()
        self.models         = models or ["gpt-4o-mini"]
        self.scorer         = scorer
        self.primary_metric = primary_metric

    def compare(
        self,
        prompt_a: str,
        prompt_b: str,
        inputs: list,
        prompt_a_id: str = "prompt_a",
        prompt_b_id: str = "prompt_b",
    ) -> dict:
        """
        Run A/B comparison across all configured models.
        Returns dict of { model: ABReport }.
        """
        from .runner import BatchRunner, render_prompt

        reports = {}

        for model in self.models:
            logger.info(f"\n  A/B: {prompt_a_id} vs {prompt_b_id} on {model}")
            a_scores_all = {metric: [] for metric in [self.primary_metric]}
            b_scores_all = {metric: [] for metric in [self.primary_metric]}
            all_metrics = set()

            for inp in inputs:
                inp_id = inp.get("id", "?")
                rendered_a = render_prompt(prompt_a, inp)
                rendered_b = render_prompt(prompt_b, inp)

                result_a = self.client.call(model=model, prompt=rendered_a)
                result_b = self.client.call(model=model, prompt=rendered_b)

                scores_a = {}
                scores_b = {}

                if self.scorer and not result_a.error:
                    scores_a = self.scorer.score(result_a.output, inp, rendered_a)
                if self.scorer and not result_b.error:
                    scores_b = self.scorer.score(result_b.output, inp, rendered_b)

                # Also add latency as a metric
                scores_a["latency_s"] = result_a.latency_s
                scores_b["latency_s"] = result_b.latency_s

                all_metrics.update(scores_a.keys())
                all_metrics.update(scores_b.keys())

                for metric in all_metrics:
                    if metric not in a_scores_all:
                        a_scores_all[metric] = []
                        b_scores_all[metric] = []
                    a_scores_all[metric].append(scores_a.get(metric, 0.0))
                    b_scores_all[metric].append(scores_b.get(metric, 0.0))

                logger.info(
                    f"    [{inp_id}] A={scores_a.get(self.primary_metric, '—'):.3f}"
                    f"  B={scores_b.get(self.primary_metric, '—'):.3f}"
                    if isinstance(scores_a.get(self.primary_metric), float)
                    else f"    [{inp_id}] scored"
                )

            # Build report for this model
            metrics_report = {}
            string_metrics = {"judge_rationale"}

            for metric in all_metrics:
                if metric in string_metrics:
                    continue
                a_vals = a_scores_all.get(metric, [])
                b_vals = b_scores_all.get(metric, [])
                if not a_vals or not b_vals:
                    continue

                mean_a = sum(a_vals) / len(a_vals)
                mean_b = sum(b_vals) / len(b_vals)
                delta  = mean_b - mean_a
                delta_pct = (delta / mean_a * 100) if mean_a != 0 else 0.0
                p_value = _paired_ttest_p(a_vals, b_vals)
                significant = p_value < 0.05

                winner = "tie"
                if abs(delta) > 0.001:
                    winner = "B" if delta > 0 else "A"

                metrics_report[metric] = {
                    "mean_a": round(mean_a, 4),
                    "mean_b": round(mean_b, 4),
                    "delta": round(delta, 4),
                    "delta_pct": round(delta_pct, 2),
                    "winner": winner,
                    "p_value": round(p_value, 4),
                    "significant": significant,
                }

            # Overall winner based on primary metric
            primary = metrics_report.get(self.primary_metric, {})
            overall_winner = primary.get("winner", "tie")

            if overall_winner == "B":
                rec = f"Use '{prompt_b_id}' — outperforms on {self.primary_metric}"
            elif overall_winner == "A":
                rec = f"Keep '{prompt_a_id}' — outperforms on {self.primary_metric}"
            else:
                rec = "No significant difference — consider other factors (latency, cost)"

            reports[model] = ABReport(
                prompt_a_id=prompt_a_id,
                prompt_b_id=prompt_b_id,
                model=model,
                n_inputs=len(inputs),
                metrics=metrics_report,
                overall_winner=overall_winner,
                recommendation=rec,
            )

        return reports
