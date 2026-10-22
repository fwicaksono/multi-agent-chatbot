_Run 2026-10-22 02:45 UTC · provider `openai` · model `gpt-4.1-mini` · top-k 6_

**Retrieval** (answerable questions; a hit = expected document *and* page/section)

| Agent | Mode | Hit@1 | Hit@3 | Hit@k | MRR |
|---|---|---|---|---|---|
| hr | vector | 0.93 | 1.00 | 1.00 | 0.97 |
| hr | keyword | 0.93 | 1.00 | 1.00 | 0.97 |
| hr | hybrid | 1.00 | 1.00 | 1.00 | 1.00 |
| hr_id | vector | 0.93 | 1.00 | 1.00 | 0.97 |
| hr_id | keyword | 0.87 | 1.00 | 1.00 | 0.92 |
| hr_id | hybrid | 0.93 | 1.00 | 1.00 | 0.97 |
| legal | vector | 0.93 | 1.00 | 1.00 | 0.96 |
| legal | keyword | 0.87 | 1.00 | 1.00 | 0.92 |
| legal | hybrid | 1.00 | 1.00 | 1.00 | 1.00 |

**Answers** (hybrid retrieval, end-to-end through the chat API)

| Agent | Questions | Answer acc. | Citation acc. | Abstention acc. | False IDK | Multi-turn | Avg latency | Cost |
|---|---|---|---|---|---|---|---|---|
| hr | 18 | 100% | 100% | 100% | 0% | 100% | 1.4s | $0.0081 |
| hr_id | 18 | 100% | 100% | 67% | 0% | 100% | 1.3s | $0.0089 |
| legal | 18 | 100% | 100% | 100% | 0% | 100% | 1.4s | $0.0072 |
