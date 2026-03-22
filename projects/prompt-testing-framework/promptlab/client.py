"""
promptlab/client.py
===================
Unified multi-provider API client.
Supports OpenAI, Anthropic, and OpenRouter through a single interface.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Model → provider routing table
MODEL_PROVIDER_MAP = {
    # OpenAI
    "gpt-4o":               "openai",
    "gpt-4o-mini":          "openai",
    "gpt-4-turbo":          "openai",
    "gpt-3.5-turbo":        "openai",
    # Anthropic
    "claude-opus-4-6":         "anthropic",
    "claude-sonnet-4-6":       "anthropic",
    "claude-haiku-4-5-20251001":       "anthropic",
}

# Any model with these prefixes → OpenRouter
OPENROUTER_PREFIXES = (
    "mistralai/", "meta-llama/", "google/", "microsoft/",
    "cohere/", "anthropic/", "openai/", "nousresearch/",
)


@dataclass
class CallResult:
    model: str
    provider: str
    output: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PromptLabClient:
    """
    Unified client for OpenAI, Anthropic, and OpenRouter.

    Usage:
        client = PromptLabClient()
        result = client.call("gpt-4o-mini", "Summarize this: ...")
        result = client.call("claude-haiku-4-5-20251001", "Summarize this: ...")
        result = client.call("mistralai/mistral-7b-instruct", "Summarize this: ...")
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.max_retries  = max_retries
        self.retry_delay  = retry_delay

        self._openai_key     = openai_api_key     or os.environ.get("OPENAI_API_KEY", "")
        self._anthropic_key  = anthropic_api_key  or os.environ.get("ANTHROPIC_API_KEY", "")
        self._openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")

        self._clients: dict = {}

    # ── Provider resolution ──────────────────────────────────

    def _resolve_provider(self, model: str) -> str:
        if model in MODEL_PROVIDER_MAP:
            return MODEL_PROVIDER_MAP[model]
        if any(model.startswith(p) for p in OPENROUTER_PREFIXES):
            return "openrouter"
        # Default: try OpenAI
        logger.warning(f"Unknown model '{model}' — defaulting to openai provider")
        return "openai"

    def _get_openai(self):
        if "openai" not in self._clients:
            from openai import OpenAI
            self._clients["openai"] = OpenAI(api_key=self._openai_key)
        return self._clients["openai"]

    def _get_anthropic(self):
        if "anthropic" not in self._clients:
            import anthropic
            self._clients["anthropic"] = anthropic.Anthropic(api_key=self._anthropic_key)
        return self._clients["anthropic"]

    def _get_openrouter(self):
        if "openrouter" not in self._clients:
            from openai import OpenAI
            self._clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._openrouter_key,
            )
        return self._clients["openrouter"]

    # ── Core call ────────────────────────────────────────────

    def call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CallResult:
        """
        Call any supported model with a prompt string.
        Returns a CallResult with output, latency, and token counts.
        """
        provider = self._resolve_provider(model)
        temp     = temperature if temperature is not None else self.temperature
        mtok     = max_tokens  if max_tokens  is not None else self.max_tokens

        last_error = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()

                if provider == "openai":
                    result = self._call_openai(model, prompt, system, temp, mtok)
                elif provider == "anthropic":
                    result = self._call_anthropic(model, prompt, system, temp, mtok)
                elif provider == "openrouter":
                    result = self._call_openrouter(model, prompt, system, temp, mtok)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                result.latency_s = round(time.time() - t0, 3)
                result.model     = model
                result.provider  = provider
                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for {model}: {e} — retrying")
                    time.sleep(self.retry_delay * (attempt + 1))

        return CallResult(
            model=model, provider=provider, output="",
            latency_s=0.0, error=str(last_error),
        )

    def _call_openai(self, model, prompt, system, temp, max_tokens) -> CallResult:
        client = self._get_openai()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temp, max_tokens=max_tokens,
        )
        return CallResult(
            model=model, provider="openai",
            output=resp.choices[0].message.content.strip(),
            latency_s=0.0,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )

    def _call_anthropic(self, model, prompt, system, temp, max_tokens) -> CallResult:
        client = self._get_anthropic()
        kwargs = dict(
            model=model, max_tokens=max_tokens, temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)
        return CallResult(
            model=model, provider="anthropic",
            output=resp.content[0].text.strip(),
            latency_s=0.0,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def _call_openrouter(self, model, prompt, system, temp, max_tokens) -> CallResult:
        client = self._get_openrouter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temp, max_tokens=max_tokens,
        )
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return CallResult(
            model=model, provider="openrouter",
            output=resp.choices[0].message.content.strip(),
            latency_s=0.0,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )

    # ── Batch call ───────────────────────────────────────────

    def call_many(
        self,
        model: str,
        prompts: list[str],
        system: Optional[str] = None,
        delay: float = 0.3,
    ) -> list[CallResult]:
        """Call the same model with a list of prompts. Returns list of CallResults."""
        results = []
        for prompt in prompts:
            result = self.call(model, prompt, system=system)
            results.append(result)
            if delay:
                time.sleep(delay)
        return results
