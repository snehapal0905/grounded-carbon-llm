# v2 (Karan) — quote-then-answer SFT → DPO, cross-family judge

A research-based extension of the main pipeline. **Status:** done. SFT + DPO trained, and evaluated head-to-head with the main model on its 150 test questions with a Gemma3-12B judge ([`results/report.md`](results/report.md)).

## What changes vs. the main pipeline
- **SFT:** quote-then-answer format (quote the supporting passage, then answer with [S#]), answer-only loss, 2 epochs.
- **DPO** on top of the merged SFT model (sigmoid + 0.2 SFT loss, beta 0.5), with four kinds of preference pairs:
  - 200 answer-over-refusal: targets the main model's 12.2% over-refusal
  - 179 refusal-over-answer
  - 100 right-over-wrong-citation
  - 100 faithful-over-contradiction
- **Evaluation:** a judge from a different model family (Gemma3-12B, not Qwen) plus HHEM, to avoid same-family judge bias, plus AuditKIT.
- **Free-GPU support:** fp16 fixes for AlignTune/TRL on T4, TRL 1.18 fused LM-head DPO to fit in 16 GB, and a resumable 3-stage Kaggle run (`kaggle/karan_v2_kaggle.ipynb`, `run_all.sh`).

## SFT result (Kaggle T4, `results/run.log`)
| | |
|---|---|
| Model | Qwen3-4B-Instruct-2507, 4-bit LoRA r=16 (33M trainable, 0.81%) |
| Data | 738 examples: 589 grounded, 110 hard negatives (gold passage removed), 69 unanswerable |
| Training | 2 epochs, 94 steps, 4.05 h, peak 7.6 GB |
| Train loss | 2.30 → 1.02 |
| Token accuracy | 0.58 → 0.75 |

## DPO result (Kaggle T4)
579 pairs, 1 epoch, 37 steps, 2.3 h, peak 10.2 GB. Loss 0.84 → 0.22, reward margin 0 → 4.5, reward accuracy 1.0.

## Test result: same 150 questions as the main pipeline, Gemma3-12B judge
| | Base + grounding prompt | v2: SFT | v2: SFT + DPO | Main: fine-tuned |
|---|---|---|---|---|
| Did the right thing ↑ | 97.3 | 96.7 | 95.3 | **98.0** |
| Grounded-refusal F1 ↑ | 89.3 | **91.3** | **91.3** | 90.0 |
| Hallucination among answered ↓ | 1.2 | **0.0** | 1.3 | **0.0** |
| Answered an unanswerable ↓ | **0.0** | **0.0** | **0.0** | **0.0** |
| Over-refusal on answerable ↓ | **11.1** | 14.4 | 16.7 | 12.2 |
| Strict accuracy on answerable ↑ | 70.0 | 66.7 | 65.6 | **73.3** |
| Cites the gold passage ↑ | 85.1 | **88.1** | 87.9 | n/a |
| Quote copied verbatim ↑ | 76.6 | **88.3** | 85.3 | – |

The main model is the best overall. v2 SFT ties it on hallucination (0%) and leads on verifiable grounding: F1, gold citations, and verbatim quotes. DPO did not cut over-refusal; the pair mix needs rebalancing. No fine-tuned system differs significantly from the grounding-prompt baseline (McNemar p ≥ 0.45).

## Running it
These files replace their counterparts in the repo root (the full code is on branch `karan-v2`). Run `kaggle/karan_v2_kaggle.ipynb` on Kaggle with `RUN = 1, 2, 3` (SFT, then DPO, then answer + score).
