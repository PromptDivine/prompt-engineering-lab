"""
retriever.py
============
Grounded QA — Lightweight TF-IDF Retriever
Project: P5 · prompt-engineering-lab

No external vector DB or embeddings API required.
Uses pure Python TF-IDF cosine similarity for passage retrieval.

Usage:
    retriever = Retriever(docs_dir="data/documents/")
    retriever.index()

    results = retriever.retrieve("What is the EU AI Act fine for violations?", top_k=3)
    for chunk in results:
        print(chunk.score, chunk.text[:100])
"""

import re
import math
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    doc_name: str
    chunk_id: str
    text: str
    score: float


class Retriever:
    """
    TF-IDF retriever over a directory of .txt documents.

    Documents are chunked into overlapping passages, indexed,
    and retrieved by cosine similarity.
    """

    def __init__(
        self,
        docs_dir: str = "data/documents",
        chunk_size: int = 150,      # words per chunk
        chunk_overlap: int = 30,    # word overlap between chunks
    ):
        self.docs_dir     = Path(docs_dir)
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[RetrievedChunk] = []
        self._tfidf: list[dict] = []   # TF-IDF vectors per chunk
        self._idf: dict = {}           # IDF scores per term
        self._indexed = False

    # ── Tokenization ─────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very common stop words
        stops = {
            'the','a','an','and','or','but','in','on','at','to','for','of',
            'with','by','from','is','are','was','were','be','been','has',
            'have','had','it','its','this','that','these','those','as','not',
            'also','which','who','what','when','where','how','than','then'
        }
        return [t for t in tokens if t not in stops and len(t) > 1]

    # ── Chunking ─────────────────────────────────────────────

    def _chunk_document(self, text: str, doc_name: str) -> list:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:
                continue
            chunk = RetrievedChunk(
                doc_name=doc_name,
                chunk_id=f"{doc_name}_{i}",
                text=" ".join(chunk_words),
                score=0.0,
            )
            chunks.append(chunk)
        return chunks

    # ── Indexing ─────────────────────────────────────────────

    def index(self):
        """Load all .txt files, chunk, and build TF-IDF index."""
        txt_files = list(self.docs_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.docs_dir}")

        all_chunks = []
        for path in txt_files:
            text = path.read_text(encoding="utf-8")
            chunks = self._chunk_document(text, path.stem)
            all_chunks.extend(chunks)
            logger.info(f"  Indexed {path.stem}: {len(chunks)} chunks")

        self.chunks = all_chunks
        n_docs = len(all_chunks)

        # Compute TF for each chunk
        tf_vectors = []
        df = {}  # document frequency per term
        for chunk in all_chunks:
            tokens = self._tokenize(chunk.text)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # Normalize TF
            total = sum(tf.values())
            tf = {t: c / total for t, c in tf.items()}
            tf_vectors.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1

        # Compute IDF
        self._idf = {
            t: math.log(n_docs / (1 + df[t]))
            for t in df
        }

        # Compute TF-IDF vectors
        self._tfidf = [
            {t: tf_val * self._idf.get(t, 0) for t, tf_val in tf.items()}
            for tf in tf_vectors
        ]

        self._indexed = True
        logger.info(f"  Index built: {n_docs} chunks, {len(self._idf)} unique terms")

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the top_k most relevant chunks for a query.
        Returns list of RetrievedChunk sorted by score descending.
        """
        if not self._indexed:
            self.index()

        query_tokens = self._tokenize(query)
        query_tf = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1
        total = sum(query_tf.values()) or 1
        query_vec = {
            t: (c / total) * self._idf.get(t, 0)
            for t, c in query_tf.items()
        }

        scored = []
        for i, chunk_vec in enumerate(self._tfidf):
            score = self._cosine(query_vec, chunk_vec)
            scored.append((score, i))

        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                doc_name=chunk.doc_name,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=round(score, 4),
            ))
        return results

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
        norm_a = math.sqrt(sum(v**2 for v in a.values()))
        norm_b = math.sqrt(sum(v**2 for v in b.values()))
        return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0

    def retrieve_as_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve top_k chunks and format as a single context string
        ready for injection into a prompt.
        """
        chunks = self.retrieve(query, top_k=top_k)
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk.doc_name}]\n{chunk.text}")
        return "\n\n".join(parts)


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = Retriever(docs_dir="data/documents")
    r.index()

    queries = [
        "What is the EU AI Act penalty for violations?",
        "How efficient are the new sodium ion batteries?",
        "What temperature records were broken in 2023?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = r.retrieve(q, top_k=2)
        for chunk in results:
            print(f"  [{chunk.score:.3f}] {chunk.doc_name}: {chunk.text[:120]}...")
