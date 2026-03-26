"""
utils/evaluation_tools.py
==========================
Prompt Engineering Lab — Shared Evaluation Utilities

Centralizes evaluation functions that were previously duplicated
across P1 (summarization-benchmark), P2 (style-transfer-prompts),
P3 (instruction-following), P6 (email-summarizer), and P9.

Usage from any project:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
    from utils.evaluation_tools import compute_rouge, compute_fk_grade, tokenize

Or with the repo root on your PYTHONPATH:
    from utils.evaluation_tools import compute_rouge
"""

import re
import math


# ──────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Tokenize text into lowercase word tokens, stripping punctuation.

    Example:
        tokenize("Hello, World!") → ["hello", "world"]
    """
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_alpha(text: str, min_len: int = 2, remove_stopwords: bool = False) -> list:
    """
    Tokenize to alphabetic words only, with optional stopword removal.
    Used by TF-IDF retrieval in P5, P8, P9.
    """
    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","by","from","is","are","was","were","be","been","has",
        "have","had","it","its","this","that","as","not","also","which",
        "he","she","they","we","i","you","its","our","their","my","your",
    }
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    tokens = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ──────────────────────────────────────────────
# ROUGE
# ──────────────────────────────────────────────

def _ngrams(tokens: list, n: int) -> dict:
    """Count n-gram occurrences in a token list."""
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _rouge_n_score(hypothesis: str, reference: str, n: int) -> dict:
    """Compute ROUGE-N precision, recall, and F1."""
    h_tokens = tokenize(hypothesis)
    r_tokens = tokenize(reference)
    h_grams  = _ngrams(h_tokens, n)
    r_grams  = _ngrams(r_tokens, n)

    overlap   = sum(min(h_grams.get(g, 0), r_grams[g]) for g in r_grams)
    r_total   = sum(r_grams.values())
    h_total   = sum(h_grams.values())

    recall    = overlap / r_total if r_total else 0.0
    precision = overlap / h_total if h_total else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _lcs_length(x: list, y: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if not m or not n:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if x[i-1] == y[j-1] else max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l_score(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-L (longest common subsequence) precision, recall, F1."""
    h_tokens = tokenize(hypothesis)
    r_tokens = tokenize(reference)
    lcs      = _lcs_length(h_tokens, r_tokens)

    recall    = lcs / len(r_tokens) if r_tokens else 0.0
    precision = lcs / len(h_tokens) if h_tokens else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        hypothesis: The generated text to evaluate
        reference:  The gold-standard reference text

    Returns:
        dict with keys: rouge1, rouge2, rougeL  (all F1 scores, 0–1)

    Example:
        scores = compute_rouge("The cat sat on the mat.", "The cat is on the mat.")
        # {'rouge1': 0.833, 'rouge2': 0.667, 'rougeL': 0.833}
    """
    return {
        "rouge1": _rouge_n_score(hypothesis, reference, 1)["f1"],
        "rouge2": _rouge_n_score(hypothesis, reference, 2)["f1"],
        "rougeL": _rouge_l_score(hypothesis, reference)["f1"],
    }


def compute_rouge_full(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L with precision/recall/F1 for each.
    Returns the full breakdown (used in detailed evaluation reports).
    """
    return {
        "rouge1": _rouge_n_score(hypothesis, reference, 1),
        "rouge2": _rouge_n_score(hypothesis, reference, 2),
        "rougeL": _rouge_l_score(hypothesis, reference),
    }


# ──────────────────────────────────────────────
# Readability
# ──────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip("'.,!?;:")
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and len(word) > 2:
        count = max(count - 1, 1)
    return max(1, count)


def compute_fk_grade(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade Level.

    Returns US school grade level (e.g., 8.0 = 8th grade).
    Higher = more complex. General public target: 6–9. Academic: 12–16.

    Example:
        compute_fk_grade("The cat sat on the mat.") → ~2.1
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words     = tokenize(text)

    if not sentences or not words:
        return 0.0

    asl = len(words) / len(sentences)
    asw = sum(_count_syllables(w) for w in words) / len(words)
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


def compute_flesch_reading_ease(text: str) -> float:
    """
    Compute Flesch Reading Ease score (0–100, higher = easier).
    90–100: very easy; 60–70: standard; 0–30: very difficult.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words     = tokenize(text)

    if not sentences or not words:
        return 0.0

    asl = len(words) / len(sentences)
    asw = sum(_count_syllables(w) for w in words) / len(words)
    return round(206.835 - 1.015 * asl - 84.6 * asw, 2)


# ──────────────────────────────────────────────
# Compression and length
# ──────────────────────────────────────────────

def compute_compression_ratio(original: str, summary: str) -> float:
    """
    Ratio of summary length to original length (word count).
    0.1 = 10% of original length. 1.0 = same length.

    Example:
        compute_compression_ratio("one two three four five", "one two") → 0.4
    """
    orig_words = len(tokenize(original))
    summ_words = len(tokenize(summary))
    return round(summ_words / orig_words, 4) if orig_words else 0.0


def word_count(text: str) -> int:
    """Count words in text."""
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    """Count sentences in text."""
    return len([s for s in re.split(r'[.!?]+', text) if s.strip()])


# ──────────────────────────────────────────────
# Lexical diversity
# ──────────────────────────────────────────────

def compute_ttr(text: str) -> float:
    """
    Type-Token Ratio: unique words / total words.
    Higher = more diverse vocabulary. Range: 0–1.
    """
    words = tokenize(text)
    return round(len(set(words)) / len(words), 4) if words else 0.0


# ──────────────────────────────────────────────
# Cosine similarity (TF-IDF based)
# ──────────────────────────────────────────────

def _tfidf_vector(text: str, vocab: list) -> list:
    tokens = tokenize(text)
    tf = {t: tokens.count(t) / len(tokens) for t in set(tokens)} if tokens else {}
    return [tf.get(w, 0.0) for w in vocab]


def _cosine(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0


def cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    Used as a lightweight BERTScore proxy and in retrieval.
    Range: 0–1 (1 = identical content).
    """
    vocab = list(set(tokenize(text_a + " " + text_b)))
    va = _tfidf_vector(text_a, vocab)
    vb = _tfidf_vector(text_b, vocab)
    return round(_cosine(va, vb), 4)


# ──────────────────────────────────────────────
# BERTScore (with fallback)
# ──────────────────────────────────────────────

def compute_bertscore(hypothesis: str, reference: str) -> float:
    """
    Compute BERTScore F1 using the bert_score library.
    Falls back to TF-IDF cosine similarity if not installed.

    Install for full accuracy: pip install bert-score
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference],
            lang="en", verbose=False,
            model_type="distilbert-base-uncased",
        )
        return round(float(F1[0].item()), 4)
    except ImportError:
        return cosine_similarity(hypothesis, reference)


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    hyp = "Scientists at Stanford developed cheaper sodium-ion batteries that charge 35% faster."
    ref = "Stanford researchers created sodium-ion batteries with 40% lower cost and 35% faster charging."

    print("Evaluation tools self-test\n" + "="*50)
    rouge = compute_rouge(hyp, ref)
    print(f"ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  ROUGE-L: {rouge['rougeL']}")
    print(f"FK Grade:     {compute_fk_grade(hyp)}")
    print(f"Reading Ease: {compute_flesch_reading_ease(hyp)}")
    print(f"Compression:  {compute_compression_ratio(ref, hyp)}")
    print(f"TTR:          {compute_ttr(hyp)}")
    print(f"Cosine Sim:   {cosine_similarity(hyp, ref)}")
    print(f"Word count:   {word_count(hyp)}")
    print("\n✅ All functions operational.")
