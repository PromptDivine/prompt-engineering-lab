"""
promptlab
=========
Prompt Testing Framework — Public API
Project: P4 · prompt-engineering-lab

Importable library for evaluating, comparing, and regression-testing prompts.

Quick start:
    from promptlab import PromptLab

    lab = PromptLab(models=["gpt-4o-mini", "claude-haiku-4-5-20251001"])

    # Batch eval
    results = lab.run(prompt="Summarize: {{text}}", inputs=[{"text": "..."}])

    # A/B compare
    report = lab.ab(prompt_a="...", prompt_b="...", inputs=[...])

    # Regression test
    lab.regression.save_baseline("my_prompt", prompt="...", inputs=[...])
    lab.regression.check("my_prompt", prompt="...")   # alerts on drift
"""

from .client     import PromptLabClient  # noqa: F401
from .runner     import BatchRunner, RunResult
from .scorers    import RubricScorer, LLMJudgeScorer, CompositeScorer
from .ab         import ABComparison, ABReport
from .regression import RegressionTracker
from .report     import ReportBuilder
from .lab        import PromptLab

__version__ = "0.1.0"
__all__ = [
    "PromptLab",
    "PromptLabClient",
    "BatchRunner", "RunResult",
    "RubricScorer", "LLMJudgeScorer", "CompositeScorer",
    "ABComparison", "ABReport",
    "RegressionTracker",
    "ReportBuilder",
]
