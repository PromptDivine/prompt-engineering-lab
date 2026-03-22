"""
promptlab/runner.py
===================
Batch evaluation engine.
Renders prompt templates, calls models, collects results.
"""

import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from .client import PromptLabClient, CallResult

logger = logging.getLogger(__name__)


def render_prompt(template: str, variables: dict) -> str:
    """
    Replace {{variable}} placeholders in a prompt template.

    Example:
        render_prompt("Summarize: {{text}}", {"text": "..."})
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    # Warn about unreplaced variables
    remaining = re.findall(r'\{\{(\w+)\}\}', result)
    if remaining:
        logger.warning(f"Unreplaced variables in prompt: {remaining}")
    return result


@dataclass
class RunResult:
    """Single evaluation result: one prompt × one model × one input."""
    run_id: str
    model: str
    provider: str
    prompt_id: str
    input_id: str
    rendered_prompt: str
    output: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class BatchRun:
    """Collection of RunResults from a batch evaluation."""
    run_id: str
    prompt_ids: list
    model_ids: list
    input_ids: list
    results: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        ok = sum(1 for r in self.results if not r.error)
        return ok / len(self.results)

    def filter(self, model=None, prompt_id=None, input_id=None) -> list:
        out = self.results
        if model:
            out = [r for r in out if r.model == model]
        if prompt_id:
            out = [r for r in out if r.prompt_id == prompt_id]
        if input_id:
            out = [r for r in out if r.input_id == input_id]
        return out

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.results])


class BatchRunner:
    """
    Evaluates one or more prompts across one or more models and inputs.

    Usage:
        runner = BatchRunner(
            client=PromptLabClient(),
            models=["gpt-4o-mini", "claude-haiku-4-5-20251001"],
        )

        results = runner.run(
            prompts={"v1": "Summarize: {{text}}", "v2": "TL;DR: {{text}}"},
            inputs=[{"id": "doc1", "text": "..."}, {"id": "doc2", "text": "..."}],
        )
    """

    def __init__(
        self,
        client: Optional[PromptLabClient] = None,
        models: Optional[list] = None,
        call_delay: float = 0.3,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ):
        self.client       = client or PromptLabClient(temperature=temperature, max_tokens=max_tokens)
        self.models       = models or ["gpt-4o-mini"]
        self.call_delay   = call_delay
        self.system_prompt = system_prompt

    def run(
        self,
        prompts: dict,
        inputs: list,
        scorers: Optional[list] = None,
        run_id: Optional[str] = None,
    ) -> BatchRun:
        """
        Args:
            prompts:  { prompt_id: template_string }
                      e.g. {"v1": "Summarize: {{text}}", "v2": "TL;DR: {{text}}"}
            inputs:   list of dicts, each must have an "id" key plus template variables
                      e.g. [{"id": "doc1", "text": "..."}, ...]
            scorers:  list of scorer objects (RubricScorer, LLMJudgeScorer, etc.)
            run_id:   optional label for this run

        Returns:
            BatchRun with all results
        """
        import uuid
        from datetime import datetime

        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        batch = BatchRun(
            run_id=run_id,
            prompt_ids=list(prompts.keys()),
            model_ids=self.models,
            input_ids=[inp.get("id", str(i)) for i, inp in enumerate(inputs)],
        )

        total = len(prompts) * len(inputs) * len(self.models)
        done  = 0

        for prompt_id, template in prompts.items():
            for inp in inputs:
                inp_id = inp.get("id", "unknown")
                rendered = render_prompt(template, inp)

                for model in self.models:
                    done += 1
                    logger.info(f"  [{done}/{total}] {prompt_id} × {inp_id} → {model}")

                    call_result = self.client.call(
                        model=model,
                        prompt=rendered,
                        system=self.system_prompt,
                    )

                    result = RunResult(
                        run_id=run_id,
                        model=model,
                        provider=call_result.provider,
                        prompt_id=prompt_id,
                        input_id=inp_id,
                        rendered_prompt=rendered,
                        output=call_result.output,
                        latency_s=call_result.latency_s,
                        prompt_tokens=call_result.prompt_tokens,
                        completion_tokens=call_result.completion_tokens,
                        error=call_result.error,
                        metadata={k: v for k, v in inp.items() if k != "id"},
                    )

                    # Apply scorers
                    if scorers and not call_result.error:
                        for scorer in scorers:
                            try:
                                score = scorer.score(
                                    output=call_result.output,
                                    input_vars=inp,
                                    prompt=rendered,
                                )
                                # Never store latency_s in scores — it's already a
                                # top-level RunResult field, storing it twice creates
                                # duplicate DataFrame columns downstream
                                score.pop("latency_s", None)
                                result.scores.update(score)
                            except Exception as e:
                                logger.warning(f"Scorer {scorer.__class__.__name__} failed: {e}")

                    batch.results.append(result)

                    if self.call_delay:
                        time.sleep(self.call_delay)

        logger.info(f"\n  BatchRun '{run_id}' complete: {len(batch.results)} results")
        return batch
