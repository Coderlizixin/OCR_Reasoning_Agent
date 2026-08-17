# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **OCR-Reasoning** benchmark — a comprehensive benchmark for evaluating Multimodal Large Language Models (MLLMs) on complex text-rich image reasoning tasks. It comprises 1,069 human-annotated examples spanning 6 core reasoning abilities (spatial, numerical, mathematical, enumerative, logical, multidisciplinary). The benchmark evaluates both final answers and reasoning processes.

The codebase is built on top of the **VLMEvalKit** framework (OpenCompass VLM Evaluation Kit) and is published as a research paper (arXiv 2505.17163).

## Common Commands

```bash
# Run OCR-Reasoning evaluation (single model, single dataset)
python run.py --data OCR_Reasoning --model Qwen2.5-VL-7B-Instruct --verbose

# Run with specific mode (all=infer+evaluate, infer=inference only)
python run.py --data OCR_Reasoning --model GPT4o --mode infer

# Run with custom work directory
python run.py --data OCR_Reasoning --model GPT4o --work-dir ./my_outputs

# Multi-GPU via torchrun
torchrun --nproc-per-node=1 run.py --data OCR_Reasoning --model GPT4o_20241120 --verbose
```

## Architecture

Built on **VLMEvalKit** — `run.py` is the entry point; `vlmeval/vlm/` holds local model wrappers, `vlmeval/api/` holds API wrappers (base: `OpenAIWrapper`), `vlmeval/dataset/` holds dataset classes including `OCR_Reasoning` in `image_vqa.py`. The OCR agent pipeline lives in `vlmeval/vlm/ocr_agent_v5.py` (current) and `vlmeval/vlm/pgr_ocragent.py` (V4 legacy).

### Dataset Flow
1. `run.py` → `build_dataset("OCR_Reasoning")` → auto-downloads from HuggingFace (`mx262/OCR-Reasoning`)
2. `build_prompt()` formats each sample; after inference, GPT-4 judge evaluates via `OcrR_auxeval` then `OcrR_acc`

## Environment

API keys are configured via `.env` file in the project root. The datasets auto-download to `LMUData/` (default path, configurable via `LMU_DATA` env var).