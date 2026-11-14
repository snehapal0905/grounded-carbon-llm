"""Step 3 - questions the documents do NOT answer (for refusal training + the test set).

Three kinds, all written to sound like a real analyst's question:
  missing_detail  - same topic as a passage, but asks for a detail the corpus doesn't give
  out_of_scope    - adjacent programme / jurisdiction / topic not covered by the corpus
  false_premise   - assumes a rule/number that the documents do not state

Each candidate is VERIFIED: we retrieve the top passages exactly as at test time
and an LLM judge must say they do not answer it. Only "no" survives.

Writes data/work/unanswerable.jsonl  {question, kind, split}
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from common.llm import chat, parse_json, pmap  # noqa: E402
from common.prompts import format_sources  # noqa: E402
from common.retrieval import Index, load_chunks  # noqa: E402
from config import SEED, TEST_PER_CLASS, TOP_K, VERIFY_MODEL, WORK  # noqa: E402

N_SEEDS = 60          # x 4 questions each ~= 240 candidates (fits free-tier daily limits)
TEST_SHARE = 0.4

GEN_PROMPT = """You are building a test of whether an assistant admits when its documents lack an answer.
The assistant's document collection contains these documents:
{docs}

Here is one passage from "{doc}":
\"\"\"{passage}\"\"\"

Write 4 realistic questions a carbon-market compliance analyst might ask that this collection
most likely does NOT answer:
- 2 of kind "missing_detail": same topic as the passage, but ask for a specific detail (a fee, a
  number, a deadline, a named entity, a procedure step) that the passage does not give.
- 1 of kind "out_of_scope": a closely related programme, country or topic NOT in the document list.
- 1 of kind "false_premise": confidently assumes a specific rule or number that is not stated.
Each question must be self-contained (name the standard or programme), under 30 words, and not
answerable with common sense alone.
Return JSON only: [{{"question": "...", "kind": "..."}}, ...]"""

VERIFY_PROMPT = """Sources:
{sources}

Question: {question}

Using ONLY these sources, can the question be answered? Say "partial" if the sources give part of
the answer or enough to correct the question's premise.
Return JSON only: {{"answerable": "yes" | "partial" | "no"}}"""


def main():
    chunks = load_chunks()
    index = Index(chunks)
    rng = random.Random(SEED)
    docs = sorted({c["doc"] for c in chunks})
    seeds = rng.sample(chunks, min(N_SEEDS, len(chunks)))

    def gen(c):
        txt = chat(VERIFY_MODEL, [{"role": "user", "content": GEN_PROMPT.format(
            docs="\n".join(f"- {d}" for d in docs), doc=c["doc"], passage=c["text"][:1500])}],
            temperature=0.8)
        return [q for q in (parse_json(txt) or []) if isinstance(q, dict) and q.get("question")]

    cands, seen = [], set()
    for qs in pmap(gen, seeds, desc="generate"):
        for q in qs or []:
            key = q["question"].lower().strip()
            if key not in seen:
                seen.add(key)
                cands.append({"question": q["question"].strip(), "kind": q.get("kind", "missing_detail")})
    print(f"{len(cands)} candidates")

    def verify(q):
        ctx = index.search(q["question"], k=TOP_K)
        j = parse_json(chat(VERIFY_MODEL, [{"role": "user", "content": VERIFY_PROMPT.format(
            sources=format_sources(ctx), question=q["question"])}], max_tokens=600))
        return (j or {}).get("answerable")

    verdicts = pmap(verify, cands, desc="verify")
    kept = [q for q, v in zip(cands, verdicts) if v == "no"]
    print(f"{len(kept)} verified unanswerable ({sum(v == 'yes' for v in verdicts)} were answerable, "
          f"{sum(v == 'partial' for v in verdicts)} partial)")

    # Split; then drop train questions that are near-duplicates of a test question.
    rng.shuffle(kept)
    n_test = max(min(TEST_PER_CLASS, int(len(kept) * TEST_SHARE)), 1)
    test, train = kept[:n_test], kept[n_test:]
    if test and train:
        tv = index.model.encode([q["question"] for q in test], normalize_embeddings=True)
        trv = index.model.encode([q["question"] for q in train], normalize_embeddings=True)
        sims = (trv @ tv.T).max(axis=1)
        train = [q for q, s in zip(train, sims) if s < 0.88]
    with open(WORK / "unanswerable.jsonl", "w", encoding="utf-8") as f:
        for split, qs in (("test", test), ("train", train)):
            for q in qs:
                f.write(json.dumps({**q, "split": split}, ensure_ascii=False) + "\n")
    stats = {"candidates": len(cands), "verified_no": len(kept), "test": len(test), "train": len(train),
             "kinds": {k: int(np.sum([q["kind"] == k for q in kept])) for k in
                       ("missing_detail", "out_of_scope", "false_premise")}}
    json.dump(stats, open(WORK / "unanswerable_stats.json", "w"), indent=2)
    print(stats)


if __name__ == "__main__":
    main()
