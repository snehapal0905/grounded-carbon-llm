"""Step 4 - build the SFT training set and the test-set candidates.

Training examples look exactly like inference: system prompt + 4 retrieved sources + question.
  positive       : gold passage is among the sources -> answer + [S#] citation
  hard negative  : SAME question, gold passage (and anything containing the answer) removed,
                   the rest retrieved normally -> "Not found in the provided documents."
  unanswerable   : verified out-of-corpus question + its real retrieved sources -> refusal
Contrasting the same question with/without its evidence is what teaches "answer only if it's there".
Only train-split chunks are ever used as training context.

Outputs
  outputs/sft/train.jsonl, outputs/sft/validation.jsonl   (messages format, AlignTune-ready)
  outputs/test_candidates.csv     <- HAND-CHECK THIS: set keep=0 on bad rows
"""
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.prompts import REFUSAL, build_messages  # noqa: E402
from common.retrieval import Index, answer_recall, is_neighbor, load_chunks  # noqa: E402
from config import NEGATIVE_RATE, OUT, SEED, TEST_PER_CLASS, TOP_K, WORK  # noqa: E402

BAD_Q = re.compile(r"\b(the|this|above|given|provided) (passage|section|text|excerpt|paragraph)\b|"
                   r"according to the (passage|text)", re.I)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    rng = random.Random(SEED)
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    index = Index(chunks)
    train_mask = index.mask(splits={"train"})

    qa = read_jsonl(WORK / "qa_raw.jsonl")
    unans = read_jsonl(WORK / "unanswerable.jsonl")
    n0 = len(qa)
    qa = [r for r in qa if not BAD_Q.search(r["question"]) and len(r["answer"]) > 2]
    print(f"QA pairs: {n0} -> {len(qa)} after dropping non-self-contained questions")

    examples, stats = [], Counter()
    for r in qa:
        if r["split"] != "train":
            continue
        gold = by_id[r["chunk_id"]]
        # positive: gold + top distractors (no neighbours of gold, so the label is unambiguous)
        cands = index.search(r["question"], k=TOP_K + 6, mask=train_mask)
        distract = [c for c in cands if c["chunk_id"] != gold["chunk_id"] and not is_neighbor(c, gold)]
        ctx = distract[:TOP_K - 1] + [gold]
        rng.shuffle(ctx)
        pos = ctx.index(gold) + 1
        examples.append({"messages": build_messages(r["question"], ctx,
                                                    answer=f"{r['answer'].rstrip()} [S{pos}]"),
                         "type": "positive"})
        stats["positive"] += 1
        # hard negative: same question, evidence removed
        if rng.random() < NEGATIVE_RATE:
            neg = [c for c in cands if c["chunk_id"] != gold["chunk_id"] and not is_neighbor(c, gold)
                   and answer_recall(r["answer"], c["text"]) < 0.5][:TOP_K]
            if len(neg) == TOP_K:
                examples.append({"messages": build_messages(r["question"], neg, answer=REFUSAL),
                                 "type": "hard_negative"})
                stats["hard_negative"] += 1
            else:
                stats["hard_negative_skipped_leak"] += 1
    for q in unans:
        if q["split"] != "train":
            continue
        ctx = index.search(q["question"], k=TOP_K, mask=train_mask)
        examples.append({"messages": build_messages(q["question"], ctx, answer=REFUSAL),
                         "type": f"unanswerable_{q['kind']}"})
        stats["unanswerable"] += 1

    rng.shuffle(examples)
    n_val = max(int(len(examples) * 0.05), 20)
    sft = OUT / "sft"
    sft.mkdir(parents=True, exist_ok=True)
    for name, part in (("validation", examples[:n_val]), ("train", examples[n_val:])):
        with open(sft / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in part:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
    refusal_share = sum(e["messages"][-1]["content"] == REFUSAL for e in examples) / len(examples)
    print(f"SFT: {len(examples) - n_val} train / {n_val} val  {dict(stats)}  refusal share {refusal_share:.0%}")

    # ---- test candidates (hand-check in Excel / Sheets) ----
    test_qa = [r for r in qa if r["split"] == "test"]
    rng.shuffle(test_qa)
    per_chunk, picked = Counter(), []
    for r in test_qa:  # at most 2 questions per chunk -> broader coverage
        if per_chunk[r["chunk_id"]] < 2:
            per_chunk[r["chunk_id"]] += 1
            picked.append(r)
    picked = picked[:TEST_PER_CLASS]
    rows = [{"id": f"A{i:03d}", "type": "answerable", "kind": "", "question": r["question"],
             "gold_answer": r["answer"], "gold_chunk_id": r["chunk_id"],
             "source_text": by_id[r["chunk_id"]]["text"], "keep": 1} for i, r in enumerate(picked)]
    rows += [{"id": f"U{i:03d}", "type": "unanswerable", "kind": q["kind"], "question": q["question"],
              "gold_answer": REFUSAL, "gold_chunk_id": "", "source_text": "", "keep": 1}
             for i, q in enumerate([q for q in unans if q["split"] == "test"][:TEST_PER_CLASS])]
    path = OUT / "test_candidates.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Test candidates: {sum(r['type'] == 'answerable' for r in rows)} answerable, "
          f"{sum(r['type'] == 'unanswerable' for r in rows)} unanswerable -> {path}")
    print("NEXT: open the CSV, read each row, set keep=0 for wrong/ambiguous ones (target >=75 each).")
    json.dump({"sft_train": len(examples) - n_val, "sft_val": n_val, **stats,
               "refusal_share": refusal_share}, open(WORK / "sft_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
