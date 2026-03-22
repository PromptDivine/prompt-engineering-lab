"""
promptlab/lab.py
================
PromptLab — main facade class.
Single entry point that wires together all components.

Usage:
    from promptlab import PromptLab

    lab = PromptLab(models=["gpt-4o-mini", "claude-haiku-4-5-20251001"])

    # 1. Batch evaluation
    batch = lab.run(
        prompts={"v1": "Summarize: {{text}}", "v2": "TL;DR: {{text}}"},
        inputs=[{"id": "doc1", "text": "..."}],
        checks=[PromptLab.no_refusal(), PromptLab.word_limit(100)],
    )
    batch.report.print_summary()

    # 2. A/B comparison
    reports = lab.ab(
        prompt_a="Summarize briefly: {{text}}",
        prompt_b="Give a 2-sentence summary: {{text}}",
        inputs=[...],
    )

    # 3. Regression testing
    lab.regression.save_baseline("my_prompt", prompt="...", inputs=[...])
    report = lab.regression.check("my_prompt", prompt="...", inputs=[...])
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PromptLab:
    """
    Main entry point for the promptlab framework.

    Args:
        models:       List of model IDs to test against.
        temperature:  Sampling temperature (default 0.3).
        max_tokens:   Max output tokens (default 1000).
        baselines_dir: Directory for regression baselines.
        call_delay:   Seconds between API calls (rate limiting).
    """

    def __init__(
        self,
        models: Optional[list] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        baselines_dir: str = "baselines",
        call_delay: float = 0.3,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        from .client     import PromptLabClient
        from .runner     import BatchRunner
        from .ab         import ABComparison
        from .regression import RegressionTracker

        self.models = models or ["gpt-4o-mini"]

        self._client = PromptLabClient(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            openrouter_api_key=openrouter_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._runner = BatchRunner(
            client=self._client,
            models=self.models,
            call_delay=call_delay,
        )

        self._ab = ABComparison(
            client=self._client,
            models=self.models,
        )

        self.regression = RegressionTracker(baselines_dir=baselines_dir)

    # ── Batch evaluation ─────────────────────────────────────

    def run(
        self,
        prompts: dict,
        inputs: list,
        checks: Optional[list] = None,
        llm_judge: bool = False,
        judge_model: str = "gpt-4o-mini",
        judge_dimensions: Optional[list] = None,
        run_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Run batch evaluation.

        Args:
            prompts:   { prompt_id: template }  e.g. {"v1": "Summarize: {{text}}"}
            inputs:    list of dicts with "id" key + template variables
            checks:    list of RubricCheck objects (use PromptLab.word_limit() etc.)
            llm_judge: enable LLM-as-judge scoring
            run_id:    label for this run

        Returns:
            BatchRunWrapper with .results, .report, .df
        """
        from .scorers import RubricScorer, LLMJudgeScorer, CompositeScorer
        from .report  import ReportBuilder

        scorers = []

        if checks:
            scorers.append(RubricScorer(checks=checks))

        if llm_judge:
            scorers.append(LLMJudgeScorer(
                client=self._client,
                judge_model=judge_model,
                dimensions=judge_dimensions,
            ))

        scorer = CompositeScorer(scorers) if scorers else None

        self._runner.system_prompt = system_prompt
        batch = self._runner.run(
            prompts=prompts,
            inputs=inputs,
            scorers=[scorer] if scorer else None,
            run_id=run_id,
        )

        return BatchRunWrapper(batch)

    # ── A/B comparison ───────────────────────────────────────

    def ab(
        self,
        prompt_a: str,
        prompt_b: str,
        inputs: list,
        prompt_a_id: str = "A",
        prompt_b_id: str = "B",
        checks: Optional[list] = None,
        llm_judge: bool = False,
        primary_metric: str = "rubric_score",
    ) -> dict:
        """
        A/B comparison between two prompt variants.
        Returns dict of { model: ABReport }.
        """
        from .scorers import RubricScorer, LLMJudgeScorer, CompositeScorer

        scorers = []
        if checks:
            scorers.append(RubricScorer(checks=checks))
        if llm_judge:
            scorers.append(LLMJudgeScorer(client=self._client))

        scorer = CompositeScorer(scorers) if scorers else None
        self._ab.scorer = scorer
        self._ab.primary_metric = primary_metric if scorer else "latency_s"

        return self._ab.compare(
            prompt_a=prompt_a, prompt_b=prompt_b,
            inputs=inputs,
            prompt_a_id=prompt_a_id, prompt_b_id=prompt_b_id,
        )

    # ── RubricCheck shortcuts ────────────────────────────────

    @staticmethod
    def word_limit(n: int):
        from .scorers import RubricScorer
        return RubricScorer.word_limit(n)

    @staticmethod
    def word_minimum(n: int):
        from .scorers import RubricScorer
        return RubricScorer.word_minimum(n)

    @staticmethod
    def must_contain(phrase: str):
        from .scorers import RubricScorer
        return RubricScorer.must_contain(phrase)

    @staticmethod
    def must_not_contain(phrase: str):
        from .scorers import RubricScorer
        return RubricScorer.must_not_contain(phrase)

    @staticmethod
    def no_refusal():
        from .scorers import RubricScorer
        return RubricScorer.no_refusal()

    @staticmethod
    def json_valid():
        from .scorers import RubricScorer
        return RubricScorer.json_valid()

    @staticmethod
    def numbered_list(n: int):
        from .scorers import RubricScorer
        return RubricScorer.numbered_list(n)

    @staticmethod
    def contains_pattern(pattern: str, name: str = "pattern"):
        from .scorers import RubricScorer
        return RubricScorer.contains_pattern(pattern, name)


# ── Wrapper for ergonomic access ─────────────────────────────

class BatchRunWrapper:
    """Wraps a BatchRun and provides .report, .df, .save() shortcuts."""

    def __init__(self, batch_run):
        from .report import ReportBuilder
        self._batch = batch_run
        self.report = ReportBuilder(batch_run)
        self.results = batch_run.results
        self.run_id = batch_run.run_id

    @property
    def df(self):
        return self._batch.to_dataframe()

    def save(self, results_dir: str = "results"):
        """Save CSV + JSON to results_dir/."""
        self.report.to_csv(f"{results_dir}/{self.run_id}.csv")
        self.report.to_json(f"{results_dir}/{self.run_id}.json")
        return self

    def plot(self, output_path: str = None):
        path = output_path or f"results/{self.run_id}_chart.png"
        self.report.plot(output_path=path)
        return self
