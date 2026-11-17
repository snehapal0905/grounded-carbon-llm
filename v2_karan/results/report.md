# v2 Run 3 — head-to-head on the main pipeline's 150 test questions

Kaggle T4. Same 150 questions (90 answerable + 60 unanswerable), same chunks and same 4 retrieved passages per question for every system.
Judge: Gemma3-12B via Ollama (different family from the Qwen3-4B being tested). HHEM ran out of GPU memory next to the judge and was skipped.
None of the 150 test questions appear in v2's training data.

| Metric | Base + simple prompt | Base + grounding prompt | v2: SFT | v2: SFT + DPO | Main pipeline: fine-tuned |
|---|---|---|---|---|---|
| Grounded-refusal F1 (Trust-Score style) ↑ | 81.1 | 89.3 | 91.3 | 91.3 | 90.0 |
| Did the right thing (per item) ↑ | 91.3 | 97.3 | 96.7 | 95.3 | 98.0 |
| Hallucination among answered (judge) ↓ | 1.0 | 1.2 | 0.0 | 1.3 | 0.0 |
| Responses with an unsupported claim ↓ | 0.7 | 0.7 | 0.0 | 0.7 | 0.0 |
| Correct refusal on unanswerable ↑ | 81.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| Answered an unanswerable question ↓ | 18.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Over-refusal when evidence retrieved ↓ | 0.0 | 4.3 | 4.3 | 5.7 | 4.3 |
| Over-refusal on all answerable ↓ | 3.3 | 11.1 | 14.4 | 16.7 | 12.2 |
| Accuracy on answerable (strict) ↑ | 90.0 | 70.0 | 66.7 | 65.6 | 73.3 |
| Accuracy on answerable (incl. partial) ↑ | 95.6 | 90.0 | 83.3 | 81.1 | 87.8 |
| Citation precision (cites the gold passage) ↑ | 7.1 | 85.1 | 88.1 | 87.9 | n/a* |
| Citation recall ↑ | 7.1 | 88.1 | 88.1 | 87.9 | n/a* |
| Quotes copied verbatim ↑ | - | 76.6 | 88.3 | 85.3 | - |
| Refusals using the exact sentence ↑ | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Answer rate | 65.3 | 53.3 | 51.3 | 50.0 | 52.7 |

\* The main pipeline's answers were scored from its saved predictions file, which has no gold-passage positions, so gold-citation precision/recall could not be computed for it (its own evaluation reports a 97.5% citation rate).

AuditKIT (lexical): F1 0.268 / 0.720 / 0.687 / 0.684 / 0.735; hallucination score 0.554 / 0.682 / 0.699 / 0.707 / 0.689.

**Paired vs Base + grounding prompt** (McNemar on "did the right thing"):
- Base + simple prompt: −6.0 pts (95% CI −10.7 to −1.3), p = 0.035
- v2: SFT: −0.7 pts (CI −3.3 to +2.0), p = 1
- v2: SFT + DPO: −2.0 pts (CI −5.3 to +1.3), p = 0.45
- Main pipeline: fine-tuned: +0.7 pts (CI −1.3 to +2.7), p = 1

DPO training: 579 pairs, 1 epoch, 37 steps, 2.3 h, peak 10.2 GB. Loss 0.84 → 0.22, reward margin 0 → 4.5, reward accuracy 1.0.
