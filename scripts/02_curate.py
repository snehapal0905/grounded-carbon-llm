"""Step 2 - CuratorKIT: chunks -> grounded QA pairs (quality-gated, deduplicated).

    python scripts/02_curate.py                # CuratorKIT (default)
    python scripts/02_curate.py --engine fallback   # plain Groq generator, same output schema

Writes data/work/qa_raw.jsonl  {question, answer, chunk_id, split, engine}
and data/work/funnel.json (counts for the README data-funnel chart).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (GATE_MODEL, GEN_MODEL, LLM_CONCURRENCY, MAX_TEST_CHUNKS,  # noqa: E402
                    MAX_TRAIN_CHUNKS, QUESTIONS_PER_CHUNK, SEED, WORK)
from common.retrieval import load_chunks  # noqa: E402


def seed_text(c):
    # The document name goes into the passage so generated questions can name the
    # standard ("Under the VCS Standard, ...") instead of "according to the passage".
    return f"Document: {c['doc']}\n\n{c['text']}"


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:400]


def select_chunks(chunks):
    rng = random.Random(SEED)
    train = [c for c in chunks if c["split"] == "train"]
    test = [c for c in chunks if c["split"] == "test"]
    rng.shuffle(train)
    rng.shuffle(test)
    return train[:MAX_TRAIN_CHUNKS] + test[:MAX_TEST_CHUNKS]


def run_curatorkit(selected):
    import os
    from curatorkit import Curator, CuratorConfig
    TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

    inp = WORK / "curator_input.jsonl"
    with open(inp, "w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps({"text": seed_text(c), "source_file": c["doc"],
                                "page": c["page"], "chunk_index": c["n"]}, ensure_ascii=False) + "\n")
    out_dir = WORK / "curator_out"
    result = Curator(CuratorConfig(
        dataset=str(inp),
        llm_model=GEN_MODEL,
        judge_llm_model=GATE_MODEL,           # different model family judges the generator
        generation_task="qa",
        num_questions=QUESTIONS_PER_CHUNK,
        difficulty="medium",
        hallucination_threshold=0.7,          # reject answers not grounded in the source chunk
        dedup="minhash",
        clean=True,
        export_formats=["alpaca"],
        llm_concurrency=LLM_CONCURRENCY,
        # many retries (API 429s back off); long timeout for a local GPU model under load
        generator_llm={"max_retries": 12, "timeout": TIMEOUT, "temperature": 0.4, "max_tokens": 2000,
                       **({"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in GEN_MODEL else {})},
        hallucination_llm={"model": GATE_MODEL, "temperature": 0.0, "max_tokens": 400,
                           "max_retries": 12, "timeout": TIMEOUT},
        output_dir=str(out_dir),
    )).run()
    result.print_summary()
    rows = [{"question": s.instruction, "answer": s.output, "passage": s.input,
             "meta": s.metadata} for s in result.passed]
    funnel = {
        "engine": "curatorkit",
        "seed_chunks": len(selected),
        "stage_counts": result.stage_counts,
        "rejected_by_step": dict(Counter(r.rejecting_step for r in result.rejected)),
        "passed": len(result.passed),
    }
    return rows, funnel


QA_PROMPT = """You write training data for a carbon-market compliance assistant.
Read the passage and write {n} question-answer pairs.
Rules:
- Each question must be answerable ONLY from this passage, and must be self-contained: name the
  standard/programme/document it is about (never say "the passage", "this section", "the text").
- Answers: 1-3 sentences, copy facts exactly (numbers, dates, defined terms). No outside knowledge.
- Prefer specific, checkable facts: requirements, thresholds, durations, definitions, who-does-what.
Return JSON only: [{{"question": "...", "answer": "..."}}, ...]

PASSAGE:
{passage}"""

GROUND_PROMPT = """Passage:
{passage}

Question: {question}
Answer: {answer}

Is every fact in the answer directly supported by the passage, and does it answer the question?
Return JSON only: {{"score": <0.0-1.0>, "reason": "<short>"}}"""


def run_fallback(selected):
    from common.llm import chat, parse_json, pmap

    def gen(c):
        txt = chat(GEN_MODEL, [{"role": "user", "content": QA_PROMPT.format(
            n=QUESTIONS_PER_CHUNK, passage=seed_text(c))}], temperature=0.4)
        pairs = parse_json(txt) or []
        return [{"question": p["question"], "answer": p["answer"], "passage": seed_text(c),
                 "chunk_id": c["chunk_id"]} for p in pairs if isinstance(p, dict) and p.get("question")]

    generated = [r for rs in pmap(gen, selected, desc="generate") if rs for r in rs]

    def gate(r):
        j = parse_json(chat(GATE_MODEL, [{"role": "user", "content": GROUND_PROMPT.format(**r)}]))
        return float((j or {}).get("score", 0))

    scores = pmap(gate, generated, desc="hallucination gate")
    passed = [r for r, s in zip(generated, scores) if s is not None and s >= 0.7]
    seen, deduped = set(), []
    for r in passed:
        k = norm(r["question"])[:120]
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    funnel = {"engine": "fallback", "seed_chunks": len(selected), "generated": len(generated),
              "after_hallucination_gate": len(passed), "after_dedup": len(deduped),
              "passed": len(deduped)}
    return deduped, funnel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["curatorkit", "fallback"], default="curatorkit")
    args = ap.parse_args()

    chunks = load_chunks()
    selected = select_chunks(chunks)
    print(f"Generating from {len(selected)} chunks with {GEN_MODEL}; gate {GATE_MODEL}")
    rows, funnel = run_curatorkit(selected) if args.engine == "curatorkit" else run_fallback(selected)

    # Map every QA pair back to its source chunk (needed for the train/test split
    # and to know which passage is the "gold" one).
    by_text = {norm(seed_text(c)): c for c in selected}
    by_id = {c["chunk_id"]: c for c in chunks}
    index = None
    out, unmapped = [], 0
    for r in rows:
        c = by_id.get(r.get("chunk_id")) or by_text.get(norm(r["passage"]))
        if c is None:
            if index is None:
                from common.retrieval import Index
                index = Index(selected)
            c = index.nearest(r["passage"])
            unmapped += 1
        out.append({"question": r["question"].strip(), "answer": r["answer"].strip(),
                    "chunk_id": c["chunk_id"], "split": c["split"], "engine": funnel["engine"]})
    with open(WORK / "qa_raw.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    funnel["mapped_by_embedding"] = unmapped
    funnel["by_split"] = dict(Counter(r["split"] for r in out))
    json.dump(funnel, open(WORK / "funnel.json", "w"), indent=2, default=str)
    print(json.dumps(funnel, indent=2, default=str))


if __name__ == "__main__":
    main()
