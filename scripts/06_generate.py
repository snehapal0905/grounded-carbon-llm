"""Step 6 - run the 3 systems on the hand-checked test set (Kaggle GPU).

  base_rag     : base model + retrieval + simple prompt
  base_strict  : base model + retrieval + the same strict prompt used in training
  tuned_rag    : fine-tuned model + retrieval + strict prompt

All three see the IDENTICAL retrieved passages per question, so differences come from the model.
Writes outputs/preds/<system>.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model import Generator  # noqa: E402
from common.prompts import SYSTEM_SIMPLE, SYSTEM_STRICT, build_messages  # noqa: E402
from common.retrieval import Index, is_neighbor, load_chunks  # noqa: E402
from config import OUT, TOP_K  # noqa: E402

SYSTEMS = {"base_rag": (SYSTEM_SIMPLE, False),
           "base_strict": (SYSTEM_STRICT, False),
           "tuned_rag": (SYSTEM_STRICT, True)}


def load_test():
    with open(OUT / "test_candidates.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("keep", "1")).strip() not in ("0", "")]
    print(f"Test set: {sum(r['type'] == 'answerable' for r in rows)} answerable, "
          f"{sum(r['type'] == 'unanswerable' for r in rows)} unanswerable")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_test()[:args.limit]
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    index = Index(chunks)
    for r in rows:
        ctx = index.search(r["question"], k=TOP_K)
        r["ctx"] = ctx
        gold = by_id.get(r["gold_chunk_id"])
        r["retrieval_hit"] = bool(gold) and any(c["chunk_id"] == gold["chunk_id"] or is_neighbor(c, gold)
                                                for c in ctx)
    ans = [r for r in rows if r["type"] == "answerable"]
    print(f"Retrieval recall@{TOP_K} (gold or neighbour chunk): "
          f"{sum(r['retrieval_hit'] for r in ans) / max(len(ans), 1):.1%}")

    gen = Generator()
    (OUT / "preds").mkdir(exist_ok=True)
    for name in args.systems.split(","):
        system, tuned = SYSTEMS[name]
        print(f"\n== {name}")
        outputs = gen.generate([build_messages(r["question"], r["ctx"], system=system) for r in rows],
                               tuned=tuned)
        with open(OUT / "preds" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outputs):
                f.write(json.dumps({
                    "id": r["id"], "type": r["type"], "kind": r["kind"], "question": r["question"],
                    "gold_answer": r["gold_answer"], "retrieval_hit": r["retrieval_hit"],
                    "ctx_ids": [c["chunk_id"] for c in r["ctx"]],
                    "ctx": [f"[S{i + 1}] {c['text']}" for i, c in enumerate(r["ctx"])],
                    "output": o}, ensure_ascii=False) + "\n")
        print(outputs[0][:300])


if __name__ == "__main__":
    main()
