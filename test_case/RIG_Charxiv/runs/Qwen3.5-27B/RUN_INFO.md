# Qwen3.5-27B RIG run snapshot

- Model used for bbox generation: `Qwen3.5-27B`
- Original run location: `test_case/RIG_Charxiv/`
- Snapshot created: 2026-08-19
- Files copied from the completed legacy run: 45
- Verification: every copied file matched its source SHA-256 hash at snapshot time
- Historical `evaluation_report.json` is preserved exactly as originally generated, but it used the mistakenly supplied 9B answer-score file and must not be cited for answer-conditioned metrics.
- `evaluation_report_corrected_27B_scores.json` is the corrected report joined by `index` to `outputs/Qwen3_5-27B-API/Qwen3_5-27B-API_CharXiv_reasoning_val_qwen-plus.xlsx`.

The presence of this file marks the directory as a protected snapshot. The RIG runner refuses both normal and `--resume` writes here; only the explicitly destructive `--overwrite-existing` option can bypass the lock.
