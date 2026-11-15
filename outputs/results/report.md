| Metric | Base + RAG | Base + strict prompt + RAG | Fine-tuned + RAG (ours) |
|---|---|---|---|
| Hallucination rate (any unsupported claim) ↓ | 17.3% | 4.7% | 1.3% |
| Correct refusal on unanswerable ↑ | 73.3% | 100.0% | 100.0% |
| Made-up answer on unanswerable ↓ | 11.7% | 0.0% | 0.0% |
| Accuracy on answerable (strict) ↑ | 26.7% | 30.0% | 50.0% |
| Accuracy on answerable (incl. partial) ↑ | 95.6% | 93.3% | 87.8% |
| Over-refusal on answerable ↓ | 3.3% | 6.7% | 12.2% |
| Over-refusal when evidence was retrieved ↓ | 0.0% | 1.4% | 4.3% |
| Answers with a source citation ↑ | 6.8% | 100.0% | 97.5% |

Test set: 90 answerable + 60 unanswerable questions (unanswerables verified against retrieval by an LLM; test chunks never seen in training). Retrieval recall@4: 77.8%. Judge: ollama_chat/qwen2.5:14b.
- Base + RAG: hallucination 95% CI [11.3%, 23.3%]
- Base + strict prompt + RAG: hallucination 95% CI [2.0%, 8.0%]
- Fine-tuned + RAG (ours): hallucination 95% CI [0.0%, 3.3%]