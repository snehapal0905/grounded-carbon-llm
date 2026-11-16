# v2 (Karan) — quote-then-answer SFT → DPO, cross-family judge

A research-based extension of the main pipeline. **Status at submission:** SFT stage trained; DPO and test-set evaluation still running. The main README's results are from the main pipeline, not this one.

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

These are training metrics. Test-set numbers will be added when DPO and evaluation finish.

## Running it
These files replace their counterparts in the repo root (the full code is on branch `karan-v2`). Run `kaggle/karan_v2_kaggle.ipynb` on Kaggle with `RUN = 1, 2, 3` (SFT, then DPO, then answer + score).
