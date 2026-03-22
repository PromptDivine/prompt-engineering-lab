"""
promptlab/regression.py
=======================
Regression testing for prompts.
Saves baseline snapshots and alerts when scores drift.

Usage:
    tracker = RegressionTracker(baselines_dir="baselines/")

    # Save a baseline (run once, commit to git)
    tracker.save_baseline(
        name="summarizer_v1",
        prompt="Summarize: {{text}}",
        inputs=[{"id": "doc1", "text": "..."}, ...],
        models=["gpt-4o-mini"],
        scorer=RubricScorer([...]),
    )

    # Check for drift (run in CI or before deploys)
    report = tracker.check(
        name="summarizer_v1",
        prompt="Summarize: {{text}}",   # new version of the prompt
    )
    if report.has_regression:
        print(report.summary())
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BaselineSnapshot:
    name: str
    prompt: str
    models: list
    created_at: str
    scores: dict          # { model: { metric: mean_value } }
    metadata: dict = field(default_factory=dict)


@dataclass
class RegressionReport:
    baseline_name: str
    baseline_date: str
    check_date: str
    regressions: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)

    # Each item: { model, metric, baseline_val, current_val, delta, delta_pct }

    @property
    def has_regression(self) -> bool:
        return len(self.regressions) > 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*62}",
            f"REGRESSION REPORT: {self.baseline_name}",
            f"Baseline: {self.baseline_date} | Checked: {self.check_date}",
            f"{'='*62}",
        ]

        if self.regressions:
            lines.append(f"\n  ⚠  REGRESSIONS ({len(self.regressions)}):")
            for r in self.regressions:
                lines.append(
                    f"     [{r['model']}] {r['metric']}: "
                    f"{r['baseline_val']:.3f} → {r['current_val']:.3f} "
                    f"({r['delta_pct']:+.1f}%)"
                )

        if self.improvements:
            lines.append(f"\n  ✓  IMPROVEMENTS ({len(self.improvements)}):")
            for r in self.improvements:
                lines.append(
                    f"     [{r['model']}] {r['metric']}: "
                    f"{r['baseline_val']:.3f} → {r['current_val']:.3f} "
                    f"({r['delta_pct']:+.1f}%)"
                )

        if self.unchanged:
            lines.append(f"\n  —  UNCHANGED: {len(self.unchanged)} metrics")

        status = "FAILED — regressions detected" if self.has_regression else "PASSED"
        lines.append(f"\n  STATUS: {status}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


class RegressionTracker:
    """
    Manages baseline snapshots and regression checks.

    Baselines are stored as JSON files in baselines_dir/.
    Commit baselines to git to track prompt performance over time.
    """

    def __init__(
        self,
        baselines_dir: str = "baselines",
        regression_threshold: float = 0.05,   # 5% drop = regression
        improvement_threshold: float = 0.05,  # 5% gain = improvement
    ):
        self.baselines_dir          = Path(baselines_dir)
        self.regression_threshold   = regression_threshold
        self.improvement_threshold  = improvement_threshold
        self.baselines_dir.mkdir(exist_ok=True)

    def _baseline_path(self, name: str) -> Path:
        safe_name = name.replace("/", "_").replace(" ", "_")
        return self.baselines_dir / f"{safe_name}.json"

    def save_baseline(
        self,
        name: str,
        prompt: str,
        inputs: list,
        models: list,
        scorer,
        client=None,
        overwrite: bool = False,
    ) -> BaselineSnapshot:
        """
        Run prompt against inputs, score outputs, save as baseline JSON.
        """
        path = self._baseline_path(name)
        if path.exists() and not overwrite:
            logger.warning(
                f"Baseline '{name}' already exists. "
                f"Pass overwrite=True to replace it."
            )
            return self.load_baseline(name)

        from .client import PromptLabClient
        from .runner import render_prompt

        client = client or PromptLabClient()
        scores_by_model = {}

        for model in models:
            logger.info(f"  Saving baseline '{name}' on {model}...")
            model_scores = []

            for inp in inputs:
                rendered = render_prompt(prompt, inp)
                result = client.call(model=model, prompt=rendered)

                if result.error:
                    logger.warning(f"  Call failed for {inp.get('id', '?')}: {result.error}")
                    continue

                s = scorer.score(result.output, inp, rendered)
                s["latency_s"] = result.latency_s
                model_scores.append(s)

            if model_scores:
                all_metrics = set(k for s in model_scores for k in s)
                scores_by_model[model] = {
                    metric: round(
                        sum(s.get(metric, 0) for s in model_scores) / len(model_scores), 4
                    )
                    for metric in all_metrics
                    if isinstance(model_scores[0].get(metric), (int, float))
                }

        snapshot = BaselineSnapshot(
            name=name,
            prompt=prompt,
            models=models,
            created_at=datetime.now().isoformat(),
            scores=scores_by_model,
        )

        with open(path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)

        logger.info(f"  Baseline saved → {path}")
        return snapshot

    def load_baseline(self, name: str) -> BaselineSnapshot:
        path = self._baseline_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"No baseline found for '{name}'. "
                f"Run save_baseline() first."
            )
        with open(path) as f:
            data = json.load(f)
        return BaselineSnapshot(**data)

    def list_baselines(self) -> list:
        return [p.stem for p in self.baselines_dir.glob("*.json")]

    def check(
        self,
        name: str,
        prompt: str,
        inputs: Optional[list] = None,
        models: Optional[list] = None,
        scorer=None,
        client=None,
    ) -> RegressionReport:
        """
        Compare current prompt performance against a saved baseline.
        Returns RegressionReport — check .has_regression for CI use.
        """
        baseline = self.load_baseline(name)

        # Use baseline settings if not overridden
        models  = models or baseline.models
        inputs  = inputs or []  # caller must provide inputs
        scorer  = scorer  # scorer must be provided

        if not inputs:
            raise ValueError("inputs must be provided to check() — same inputs used for baseline.")
        if not scorer:
            raise ValueError("scorer must be provided to check().")

        from .client import PromptLabClient
        from .runner import render_prompt

        client = client or PromptLabClient()
        current_scores = {}

        for model in models:
            model_scores = []
            for inp in inputs:
                rendered = render_prompt(prompt, inp)
                result = client.call(model=model, prompt=rendered)
                if result.error:
                    continue
                s = scorer.score(result.output, inp, rendered)
                s["latency_s"] = result.latency_s
                model_scores.append(s)

            if model_scores:
                all_metrics = set(k for s in model_scores for k in s)
                current_scores[model] = {
                    metric: round(
                        sum(s.get(metric, 0) for s in model_scores) / len(model_scores), 4
                    )
                    for metric in all_metrics
                    if isinstance(model_scores[0].get(metric), (int, float))
                }

        # Build regression report
        report = RegressionReport(
            baseline_name=name,
            baseline_date=baseline.created_at[:10],
            check_date=datetime.now().strftime("%Y-%m-%d"),
        )

        for model in models:
            baseline_model = baseline.scores.get(model, {})
            current_model  = current_scores.get(model, {})

            for metric in baseline_model:
                if metric not in current_model:
                    continue
                b_val = baseline_model[metric]
                c_val = current_model[metric]
                delta = c_val - b_val
                delta_pct = (delta / b_val * 100) if b_val != 0 else 0.0

                entry = {
                    "model": model, "metric": metric,
                    "baseline_val": b_val, "current_val": c_val,
                    "delta": round(delta, 4), "delta_pct": round(delta_pct, 2),
                }

                # For latency, higher = worse (regression)
                if metric == "latency_s":
                    if delta_pct > self.regression_threshold * 100:
                        report.regressions.append(entry)
                    elif delta_pct < -self.improvement_threshold * 100:
                        report.improvements.append(entry)
                    else:
                        report.unchanged.append(entry)
                else:
                    # For scores, lower = worse (regression)
                    if delta_pct < -self.regression_threshold * 100:
                        report.regressions.append(entry)
                    elif delta_pct > self.improvement_threshold * 100:
                        report.improvements.append(entry)
                    else:
                        report.unchanged.append(entry)

        return report
