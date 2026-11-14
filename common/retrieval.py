"""Hybrid retriever: BGE-small dense embeddings + BM25, fused with reciprocal rank fusion.

BM25 matters here: carbon-market questions often hinge on exact terms
("buffer pool", "Article 6.4", "VM0042", clause numbers) that dense models blur.
"""
import hashlib
import json
import re

import numpy as np

from config import EMB_MODEL, WORK

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(t):
    return _TOK.findall(t.lower())


def load_chunks(path=WORK / "chunks.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class Index:
    def __init__(self, chunks, device=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.ids = [c["chunk_id"] for c in chunks]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.model = SentenceTransformer(EMB_MODEL, device=device)
        h = hashlib.md5(("|".join(self.ids) + EMB_MODEL).encode()).hexdigest()[:10]
        cache = WORK / f"emb_{h}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode([c["text"] for c in chunks], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True)
            np.save(cache, self.emb)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def mask(self, splits=None, exclude=()):
        m = np.ones(len(self.ids), dtype=bool)
        if splits is not None:
            m &= np.array([c["split"] in splits for c in self.chunks])
        for cid in exclude:
            if cid in self.pos:
                m[self.pos[cid]] = False
        return m

    def search(self, query, k=4, mask=None, pool=50):
        qv = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        dense = self.emb @ qv
        sparse = self.bm25.get_scores(tokenize(query))
        if mask is not None:
            dense = np.where(mask, dense, -np.inf)
            sparse = np.where(mask, sparse, -np.inf)
        fused = {}
        for scores in (dense, sparse):
            for rank, i in enumerate(np.argsort(-scores)[:pool]):
                if np.isfinite(scores[i]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in top]

    def nearest(self, text):
        """Chunk whose text is closest to `text` (used to map generated QA back to its chunk)."""
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return self.chunks[int(np.argmax(self.emb @ v))]


def is_neighbor(a, b):
    return a["doc"] == b["doc"] and abs(a["n"] - b["n"]) <= 1


def answer_recall(answer, text):
    """Share of the answer's content words that appear in `text` (leak check for negatives)."""
    stop = {"the", "a", "an", "of", "to", "and", "or", "in", "is", "are", "be", "for", "on",
            "by", "with", "that", "this", "as", "at", "it", "from", "must", "shall", "s"}
    toks = [t for t in tokenize(answer) if t not in stop]
    if not toks:
        return 0.0
    vocab = set(tokenize(text))
    return sum(t in vocab for t in toks) / len(toks)
