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

# Run with config file (for multiple models/datasets)
python run.py --config path/to/config.json

# Multi-GPU via torchrun
torchrun --nproc-per-node=1 run.py --data OCR_Reasoning --model GPT4o_20241120 --verbose

# Run individual test scripts
python test_pgr_ocragent.py
python test_model_connection.py
python test_generate_pseudo_grounding_bbox.py
python test_maxtokens.py
python test_load_dataset.py
```

## Architecture

### Entry Point: `run.py`
Main evaluation script. Parses arguments, builds models and datasets, orchestrates inference and evaluation. Supports two modes:
- **CLI mode** (`--data`/`--model`): Uses pre-configured models from `supported_VLM` dict in `vlmeval/config.py`
- **Config mode** (`--config`): Uses a JSON config with arbitrary model/dataset class paths

### Framework: `vlmeval/`

**`vlmeval/config.py`** — Central model registry. `supported_VLM` is the master dictionary of all available models, built by merging model "series" dicts (qwen2vl_series, internvl2, cogvlm_series, api_models, etc.). Each entry is a `partial()` that creates a model instance with default parameters.

**`vlmeval/vlm/`** — Local VLM implementations (models run locally). Each file wraps a specific model family (e.g., `qwen3_vl.py` for Qwen3-VL, `internvl/` for InternVL models). `base.py` defines `BaseModel`.

**`vlmeval/api/`** — API-based model wrappers (GPT-4V, Gemini, Claude, Qwen API, etc.). `gpt.py` contains `OpenAIWrapper` — the base class for nearly all API models.

**`vlmeval/dataset/`** — Dataset classes. `image_vqa.py` contains `OCR_Reasoning` (extends `ImageBaseDataset`). `__init__.py` registers all datasets in `DATASET_CLASSES`. Each dataset class implements `build_prompt()` and `evaluate()`.

**`vlmeval/dataset/utils/ocr_reasoning.py`** — OCR-Reasoning specific evaluation logic: GPT-4 as judge with scoring rubric (1-10 scale), answer extraction, and accuracy computation.

**`vlmeval/inference.py`** — Inference pipeline for API and local models. Core function `infer_data_job()` handles the prediction loop with resume capability.

**`vlmeval/smp/`** — Shared utilities: file I/O (`load`/`dump`), logging (`get_logger`), multiprocessing helpers (`track_progress_rich`), GPU discovery.

### Key Local Changes

**`ThreeStageOCRAgent`** (`vlmeval/vlm/pgr_ocragent.py`) — A custom multi-turn OCR agent extending `OpenAIWrapper`:
1. **Stage 1 (Planning)**: Asks the model to create a step-by-step plan without calculating answers
2. **Stage 2 (Grounding)**: Asks for bounding box coordinates of relevant regions, using a separate grounding model (`Qwen3.5-122B-A10B`)
3. **Stage 3 (Review & Reasoning)**: Crops the identified regions, encodes them as base64 in-memory, and feeds them back for final reasoning

**Custom models in `supported_VLM`** (at bottom of `vlmeval/config.py`): Several Qwen3.5 models configured with local API endpoints (vLLM/sglang servers), plus `ThreeStageOCRAgent` variants (`Qwen3.5-9B-OCRAgent`, `Qwen3.5-27B-OCRAgent`).

### Dataset Flow
1. `run.py` calls `build_dataset("OCR_Reasoning")` → returns `OCR_Reasoning` instance
2. Data is auto-downloaded from HuggingFace (`mx262/OCR-Reasoning`)
3. `build_prompt()` formats each sample with language-specific instructions
4. After inference, `OCR_Reasoning.evaluate()` calls `OcrR_auxeval` (GPT-4 judge) then `OcrR_acc` (compute final scores)

## Environment

API keys are configured via `.env` file in the project root. The datasets auto-download to `LMUData/` (default path, configurable via `LMU_DATA` env var).
