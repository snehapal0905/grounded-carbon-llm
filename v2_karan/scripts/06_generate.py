"""Step 6 (v2) - run 4 systems on the test set. All see the IDENTICAL retrieved passages per question.

  base_rag     : base model + simple prompt
  base_strict  : base model + the quote-then-answer grounding prompt (strong baseline, same prompt as ours)
  tuned_sft    : base + SFT adapter            (ablation: what SFT alone gives)
  tuned_dpo    : merged SFT + DPO adapter      (final model)

Resumable: a system whose preds file already exists is skipped (use --force to redo).
TEST_CSV env var lets you score on another test set (e.g. Sneha's) for a like-for-like comparison.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model import Generator  # noqa: E402
from common.prompts import SYSTEM_QUOTE, SYSTEM_SIMPLE, build_messages, final_answer  # noqa: E402
from common.retrieval import Index, is_neighbor, load_chunks  # noqa: E402
from config import BASE_MODEL, OUT, TOP_K  # noqa: E402

MERGED_DIR = Path(os.getenv("MERGED_DIR", "/content/merged_sft"))
# system -> (prompt, model group, use adapter)
SYSTEMS = {"base_rag": (SYSTEM_SIMPLE, "base", False),
           "base_strict": (SYSTEM_QUOTE, "base", False),
           "tuned_sft": (SYSTEM_QUOTE, "base", True),
           "tuned_dpo": (SYSTEM_QUOTE, "merged", True)}


def load_test():
    path = Path(os.getenv("TEST_CSV", OUT / "test_candidates.csv"))
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("keep", "1")).strip() not in ("0", "")]
    print(f"Test set {path}: {sum(r['type'] == 'answerable' for r in rows)} answerable, "
          f"{sum(r['type'] == 'unanswerable' for r in rows)} unanswerable")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = load_test()[:args.limit]
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    index = Index(chunks)
    for r in rows:
        r["ctx"] = index.search(r["question"], k=TOP_K)
        gold = by_id.get(r["gold_chunk_id"])
        r["gold_pos"] = [i + 1 for i, c in enumerate(r["ctx"])
                         if gold and (c["chunk_id"] == gold["chunk_id"] or is_neighbor(c, gold))]
        r["retrieval_hit"] = bool(r["gold_pos"])
    ans = [r for r in rows if r["type"] == "answerable"]
    print(f"Retrieval recall@{TOP_K}: {sum(r['retrieval_hit'] for r in ans) / max(len(ans), 1):.1%}")

    pred_dir = Path(os.getenv("PREDS_DIR", OUT / "preds"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.systems.split(",") if args.force or not (pred_dir / f"{s}.jsonl").exists()]
    print("to generate:", todo or "nothing (all done)")
    for group in ("base", "merged"):
        names = [s for s in todo if SYSTEMS[s][1] == group]
        if not names:
            continue
        if group == "base":
            adapter = OUT / "adapter_sft" if any(SYSTEMS[s][2] for s in names) else None
            gen = Generator(BASE_MODEL, adapter)
        else:
            gen = Generator(MERGED_DIR, OUT / "adapter_dpo")
        for name in names:
            system, _, tuned = SYSTEMS[name]
            print(f"\n== {name}")
            outs = gen.generate([build_messages(r["question"], r["ctx"], system=system) for r in rows],
                                tuned=tuned)
            with open(pred_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({
                        "id": r["id"], "type": r["type"], "kind": r["kind"], "question": r["question"],
                        "gold_answer": r["gold_answer"], "retrieval_hit": r["retrieval_hit"],
                        "gold_pos": r["gold_pos"], "ctx_ids": [c["chunk_id"] for c in r["ctx"]],
                        "ctx": [f"[S{i + 1}] {c['text']}" for i, c in enumerate(r["ctx"])],
                        "output": o, "answer": final_answer(o)}, ensure_ascii=False) + "\n")
            print(outs[0][:400])
        gen.free()


if __name__ == "__main__":
    main()
