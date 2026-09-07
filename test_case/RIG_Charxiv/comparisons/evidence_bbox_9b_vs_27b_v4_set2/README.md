# Qwen3.5-9B vs Qwen3.5-27B evidence bbox comparison — set 2

These figures compare the two models on the same CharXiv image. Each panel uses the bbox visualization produced by `derivation-evidence-v4-concise-guided-json`; the header reports finalized semantic Recall and Precision under `match_at_0.3`.

| Index | Category | Selection role | 9B Recall / Precision | 27B Recall / Precision | Figure |
|---:|---|---|---:|---:|---|
| 914 | Text-in-Chart | overlapping-GT case; 27B improvement | 0.0% / 0.0% | 57.1% / 36.4% | `q0914_9b_vs_27b.jpg` |
| 885 | Text-in-General | 27B recovers cross-panel comparison evidence | 0.0% / 0.0% | 50.0% / 60.0% | `q0885_9b_vs_27b.jpg` |
| 547 | Number-in-Chart | 27B recovers plot label, axis and data marks | 0.0% / 0.0% | 57.1% / 36.4% | `q0547_9b_vs_27b.jpg` |
| 916 | Number-in-General | 27B improvement on subplot/legend/range evidence | 14.3% / 25.0% | 57.1% / 66.7% | `q0916_9b_vs_27b.jpg` |
| 85 | Text-in-General | counterexample where 9B is better | 50.0% / 60.0% | 16.7% / 20.0% | `q0085_9b_vs_27b.jpg` |

The source model-rendered images remain unchanged in their respective v4 run directories. These five files are side-by-side presentation copies only.

`comparison-gallery.html` provides a standalone switchable view of the five saved comparison figures.
