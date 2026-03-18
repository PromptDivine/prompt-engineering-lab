"""
evaluation.py
=============
Style Transfer Prompts — Evaluation Engine
Project: P2 · prompt-engineering-lab

Metrics computed per transfer:
  - Flesch-Kincaid Grade Level     (readability shift)
  - Lexical Diversity / TTR        (type-token ratio)
  - Formality Score                (function word ratio proxy)
  - Sentiment Polarity             (positive/negative/neutral shift)
  - Compression Ratio              (length change vs source)
  - Sentence Complexity            (avg words per sentence)
  - LLM Style Adherence Score      (judge: 1-5)
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StyleMetrics:
    fk_grade: float = 0.0
    avg_sentence_length: float = 0.0
    word_count: int = 0
    unique_word_ratio: float = 0.0
    formality_score: float = 0.0
    sentiment_polarity: float = 0.0
    compression_ratio: float = 0.0
    judge_style_adherence: float = 0.0
    judge_fluency: float = 0.0
    judge_meaning_preserved: float = 0.0
    judge_overall: float = 0.0
    judge_rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransferResult:
    source_id: str
    source_domain: str
    model: str
    prompt_id: str
    style: str
    strategy: str
    output: str
    latency_s: float
    metrics: StyleMetrics = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "source_id": self.source_id,
            "source_domain": self.source_domain,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "style": self.style,
            "strategy": self.strategy,
            "output": self.output,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }
        if self.metrics:
            d.update(self.metrics.to_dict())
        return d


def tokenize(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

def sentences(text: str) -> list:
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]

def count_syllables(word: str) -> int:
    word = word.lower().strip("'")
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and len(word) > 2:
        count -= 1
    return max(1, count)


def compute_fk_grade(text: str) -> float:
    sents = sentences(text)
    words = tokenize(text)
    if not sents or not words:
        return 0.0
    asl = len(words) / len(sents)
    asw = sum(count_syllables(w) for w in words) / len(words)
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


def compute_ttr(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


INFORMAL_MARKERS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "ok", "okay",
    "hey", "stuff", "super", "really", "pretty", "like", "just", "so",
    "totally", "literally", "basically", "actually", "honestly",
    "it's", "that's", "there's", "they're", "won't", "can't", "don't",
    "isn't", "aren't", "wasn't", "didn't", "you", "your", "we", "our"
}

FORMAL_MARKERS = {
    "shall", "herein", "pursuant", "notwithstanding", "aforementioned",
    "therefore", "however", "moreover", "furthermore", "consequently",
    "nevertheless", "subsequently", "accordingly", "whereas", "whereby",
    "approximately", "significant", "substantial", "demonstrate",
    "indicate", "facilitate", "endeavor", "utilize", "acknowledge",
    "comprising", "pertaining", "regarding", "concerning", "heretofore"
}

def compute_formality(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.5
    n = len(words)
    informal_count = sum(1 for w in words if w in INFORMAL_MARKERS)
    formal_count   = sum(1 for w in words if w in FORMAL_MARKERS)
    raw = 0.5 + (formal_count - informal_count) / (n * 0.5)
    return round(max(0.0, min(1.0, raw)), 4)


POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "positive", "benefit", "improve", "success", "effective", "strong",
    "growth", "gain", "increase", "better", "best", "outstanding",
    "significant", "valuable", "innovative", "promising", "achieve",
    "advance", "breakthrough", "opportunity", "advantage", "support"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "negative", "problem", "issue",
    "fail", "failure", "decline", "decrease", "worse", "worst", "loss",
    "damage", "risk", "threat", "concern", "difficult", "challenging",
    "crisis", "danger", "harm", "reduce", "drop", "fall", "weakness",
    "struggle", "conflict", "obstacle", "limitation", "error"
}

def compute_sentiment(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def compute_compression(source: str, output: str) -> float:
    src_words = len(tokenize(source))
    out_words = len(tokenize(output))
    return round(out_words / src_words, 4) if src_words else 0.0


STYLE_JUDGE_PROMPT = """You are an expert writing analyst evaluating style transfer quality.

TARGET STYLE: {style}
ORIGINAL TEXT: {source}
REWRITTEN TEXT: {output}

Score the rewritten text on these dimensions (1-5):
- style_adherence: How well does it match the target style? (1=not at all, 5=perfectly)
- fluency: Is the writing natural and well-crafted? (1=awkward, 5=excellent)
- meaning_preserved: Are the key facts/ideas from the original retained? (1=lost most, 5=all preserved)
- overall: Your overall quality assessment

Respond ONLY with this JSON, no other text:
{{
  "style_adherence": <score>,
  "fluency": <score>,
  "meaning_preserved": <score>,
  "overall": <average>,
  "rationale": "<one sentence>"
}}"""

def compute_llm_judge(source, output, style, client, model="gpt-4o-mini") -> dict:
    prompt = STYLE_JUDGE_PROMPT.format(style=style, source=source[:2000], output=output[:2000])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        return {
            "style_adherence":   float(data.get("style_adherence", 0)),
            "fluency":           float(data.get("fluency", 0)),
            "meaning_preserved": float(data.get("meaning_preserved", 0)),
            "overall":           float(data.get("overall", 0)),
            "rationale":         data.get("rationale", ""),
        }
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {k: 0.0 for k in ["style_adherence","fluency","meaning_preserved","overall"]}


def evaluate_transfer(
    source: str,
    output: str,
    style: str,
    judge_client=None,
    judge_model: str = "gpt-4o-mini",
    run_llm_judge: bool = False,
) -> StyleMetrics:
    m = StyleMetrics()
    words = tokenize(output)
    sents = sentences(output)

    m.fk_grade            = compute_fk_grade(output)
    m.avg_sentence_length = round(len(words) / len(sents), 2) if sents else 0.0
    m.word_count          = len(words)
    m.unique_word_ratio   = compute_ttr(output)
    m.formality_score     = compute_formality(output)
    m.sentiment_polarity  = compute_sentiment(output)
    m.compression_ratio   = compute_compression(source, output)

    if run_llm_judge and judge_client:
        j = compute_llm_judge(source, output, style, judge_client, judge_model)
        m.judge_style_adherence   = j["style_adherence"]
        m.judge_fluency           = j["fluency"]
        m.judge_meaning_preserved = j["meaning_preserved"]
        m.judge_overall           = j["overall"]
        m.judge_rationale         = j.get("rationale", "")

    return m


def compute_deltas(source_metrics: StyleMetrics, output_metrics: StyleMetrics) -> dict:
    return {
        "delta_fk_grade":        round(output_metrics.fk_grade - source_metrics.fk_grade, 2),
        "delta_formality":       round(output_metrics.formality_score - source_metrics.formality_score, 4),
        "delta_sentiment":       round(output_metrics.sentiment_polarity - source_metrics.sentiment_polarity, 4),
        "delta_sentence_length": round(output_metrics.avg_sentence_length - source_metrics.avg_sentence_length, 2),
        "delta_ttr":             round(output_metrics.unique_word_ratio - source_metrics.unique_word_ratio, 4),
    }


if __name__ == "__main__":
    source = "Our company had a really rough third quarter. Revenue dropped by 18 percent because of supply chain problems. We lost two big clients."
    formal = "The organization experienced significant performance deterioration during the third fiscal quarter. Revenue declined 18 percent, attributable primarily to supply chain disruptions. Furthermore, the departure of two major clients compounded the adverse financial impact."

    src_m = evaluate_transfer(source, source, "baseline")
    out_m = evaluate_transfer(source, formal, "academic")
    deltas = compute_deltas(src_m, out_m)

    print("\n Source:", {k: v for k, v in src_m.to_dict().items() if v})
    print("\n Output:", {k: v for k, v in out_m.to_dict().items() if v})
    print("\n Deltas:", deltas)
