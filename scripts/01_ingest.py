"""Step 1 - PDFs in data/raw/ -> data/work/chunks.jsonl

Each chunk: chunk_id, doc, n (position in doc), page, text, split.
Split is decided on blocks of consecutive chunks (not single chunks), so test
passages never leak into training through the 50-word chunk overlap. Train
chunks touching a test block become "buffer" and are used for neither.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pypdf import PdfReader  # noqa: E402

from config import CHUNK_OVERLAP, CHUNK_WORDS, RAW, SEED, WORK  # noqa: E402

TEST_BLOCK_RATE = 0.12
BLOCK = 6  # chunks per block


def clean(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)           # de-hyphenate line breaks
    text = re.sub(r"\.{4,}", " ", text)                  # table-of-contents dot leaders
    return re.sub(r"\s+", " ", text).strip()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def chunk_pdf(path):
    reader = PdfReader(str(path))
    words = []  # (word, page)
    for pno, page in enumerate(reader.pages, start=1):
        try:
            t = clean(page.extract_text() or "")
        except Exception:
            continue
        words += [(w, pno) for w in t.split()]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for start in range(0, max(len(words) - CHUNK_OVERLAP, 1), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 60:
            continue
        text = " ".join(w for w, _ in piece)
        alpha = sum(ch.isalpha() for ch in text) / max(len(text), 1)
        ascii_share = sum(ch.isascii() for ch in text) / max(len(text), 1)
        if alpha < 0.55 or ascii_share < 0.9:   # number tables, garbage, non-English (bilingual gazettes)
            continue
        chunks.append({"page": piece[0][1], "text": text})
    return chunks


def main():
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {RAW}. Download the documents listed in docs/SOURCES.md first.")
    rng = random.Random(SEED)
    out, stats = [], {}
    for pdf in pdfs:
        doc = pdf.stem
        chunks = chunk_pdf(pdf)
        n_blocks = (len(chunks) + BLOCK - 1) // BLOCK
        test_blocks = {b for b in range(n_blocks) if rng.random() < TEST_BLOCK_RATE}
        for n, c in enumerate(chunks):
            c.update(chunk_id=f"{slug(doc)}-{n:04d}", doc=doc, n=n,
                     split="test" if n // BLOCK in test_blocks else "train")
        for n, c in enumerate(chunks):  # buffer = train chunk adjacent to a test chunk
            if c["split"] == "train" and any(
                0 <= m < len(chunks) and chunks[m]["split"] == "test" for m in (n - 1, n + 1)
            ):
                c["split"] = "buffer"
        out += chunks
        stats[doc] = {s: sum(c["split"] == s for c in chunks) for s in ("train", "test", "buffer")}
        print(f"{doc[:60]:60s} {len(chunks):5d} chunks  {stats[doc]}")
    with open(WORK / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tot = {s: sum(v[s] for v in stats.values()) for s in ("train", "test", "buffer")}
    print(f"\nTotal {len(out)} chunks from {len(pdfs)} PDFs -> {tot}")
    json.dump({"per_doc": stats, "total": tot}, open(WORK / "ingest_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
