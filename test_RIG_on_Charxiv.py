"""
test_RIG_on_Charxiv.py
Reasoning Image Grounding (RIG) test on the CharXiv (reasoning) dataset.

For each question, the model outputs a JSON list of all essential chart/figure elements
with their bounding boxes (Normalized to [0, 1000]) and descriptions. The grounded
elements must be:
  - Sufficient: all information needed to answer the question is covered.
  - Minimal: only elements strictly required for deriving the answer are included.

Output format per element:
  {
    "description": "<what this element is and its role in answering>",
    "bbox": [xmin, ymin, xmax, ymax]   // all integers, 0 ≤ value ≤ 1000
  }
"""

import os
import json
import base64
import io
import re
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
#  Configuration
# ============================================================================

TSV_PATH = r"C:/Users/19639/LMUData/CharXiv_reasoning_val.tsv"
IMG_DIR = r"C:/Users/19639/LMUData/images/CharXiv_reasoning_val"
BASE_OUTPUT_DIR = Path("test_case/RIG_Charxiv")

# API  – edit these to match your deployment
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.3.112.34:23290/v1",
)

# client = OpenAI(
#     api_key="sk-Sy5R85nas3TrIuwgI3oSjg",
#     base_url="https://litellm.mybigai.ac.cn/",
# )

DEFAULT_MODEL_NAME = "Qwen3.5-27B"
MAX_TOKENS = 12288
PROMPT_VERSION = "derivation-evidence-v4-concise-guided-json"

# ============================================================================
#  Prompt templates
# ============================================================================

RIG_SYSTEM_PROMPT = """You ground visual evidence for a downstream reasoner.
Do not solve the question or choose the final answer. Return only the JSON object,
without analysis, a checklist, markdown, or the answer."""


RIG_USER_PROMPT_TEMPLATE = """IMAGE: A chart or scientific figure.

QUESTION:
__QUESTION_PLACEHOLDER__

TASK
Ground the visible evidence a downstream reasoner needs to derive the answer.
Do not ground only the final answer text or winning label.

For comparison or ranking questions, ground:
- the axis, scale, criterion, or reference used for comparison;
- the relevant mark, value, or position for every compared candidate;
- the labels or legend entries that identify the candidates.
Do not decide which candidate wins. If the wording is ambiguous, ground the visible
evidence needed to resolve it instead of debating the interpretations.

RULES
- Prefer complete evidence; slight redundancy is allowed.
- Each item is exactly one semantic clue. Separate different clues even when their
  boxes overlap or are identical.
- Use a tight box around the complete clue, not the whole subplot when a smaller
  region is enough.
- Transcribe visible text or values exactly in "description" when readable.
- bbox is [xmin, ymin, xmax, ymax], using integer coordinates in [0, 1000], with
  (0,0) at the top-left.
- Do not state the final answer.

OUTPUT ONLY this JSON structure:
{
  "elements": [
    {
      "description": "exact visible identity or transcribed text",
      "bbox": [xmin, ymin, xmax, ymax]
    }
  ]
}"""


RIG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "minItems": 1,
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 240,
                    },
                    "bbox": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1000,
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["description", "bbox"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["elements"],
    "additionalProperties": False,
}


def build_rig_extra_body() -> Dict:
    """Build vLLM request options without changing thinking or temperature."""
    return {
        "guided_json": RIG_JSON_SCHEMA,
        "mm_processor_kwargs": {
            "images_kwargs": {
                "size": {
                    "longest_edge": 8192 * 32 * 32,
                    "shortest_edge": 4096 * 32 * 32,
                }
            }
        },
    }


def build_rig_messages(prompt: str, data_uri: str) -> List[Dict]:
    return [
        {"role": "system", "content": RIG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]


# ============================================================================
#  Helpers
# ============================================================================

def load_charxiv_reasoning(tsv_path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load the CharXiv reasoning TSV.  Handles multi-line fields correctly."""
    df = pd.read_csv(tsv_path, sep="\t", nrows=n_rows)
    print(f"Loaded {len(df)} questions from {tsv_path}")
    return df


def get_rig_question(row: pd.Series) -> Tuple[str, str]:
    """Prefer the clean task question over CharXiv answer-format instructions."""
    raw_question = row.get("raw_question")
    if isinstance(raw_question, str) and raw_question.strip():
        return raw_question.strip(), "raw_question"
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip(), "question"
    raise ValueError(f"sample {row.get('index')} has no usable question text")


def get_image(row: pd.Series) -> Image.Image:
    """
    Return a PIL RGB image for a dataset row.
    Tries the local image file first, falls back to the base64 column.
    """
    idx = int(row["index"])
    local_path = Path(IMG_DIR) / f"{idx}.jpg"
    if local_path.exists():
        return Image.open(str(local_path)).convert("RGB")

    # Fallback: decode base64 stored in the TSV
    b64_str = row["image"]
    if isinstance(b64_str, str) and b64_str:
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise FileNotFoundError(f"No image found for index {idx}")


def encode_image_to_base64(img: Image.Image, max_size: int = 8192) -> Tuple[str, Tuple[int, int]]:
    """
    Encode a PIL image to a base64 data-URI string.
    Resizes if either dimension exceeds max_size ( preserves aspect ratio ).
    Returns (data_uri, (orig_w, orig_h)).
    """
    orig_size = img.size
    if img.width > max_size or img.height > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size))
        print(f"  [resize] {orig_size} -> {img.size}")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", orig_size


def parse_rig_response(text: str) -> Optional[List[Dict]]:
    """
    Parse the model's JSON response into a list of element dicts.
    Robust to reasoning text before/after the JSON block.
    Returns None on failure.
    """
    text = text.strip()

    # --- Strategy: find ALL candidate JSON objects (outermost {…}), try each ---
    candidates: List[str] = []

    # Remove markdown code fences first (keep inner content)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Collect all outermost { … } spans via balanced-brace scan
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1

    if not candidates:
        print("  [ERROR] No JSON object (balanced braces) found in model output.")
        return None

    # Try candidates in reverse order (last JSON block is most likely the intended output)
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue

        # Normalise: accept both top-level list and {"elements": [...]}
        if isinstance(data, list):
            elements = data
        elif isinstance(data, dict):
            elements = data.get("elements", [])
        else:
            continue

        if not elements:
            continue

        # Validate each element
        cleaned = []
        for i, elem in enumerate(elements):
            if not isinstance(elem, dict):
                continue
            if "bbox" not in elem:
                continue
            bbox = elem["bbox"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            # Clamp & round
            bbox_int = [max(0, min(1000, int(round(v)))) for v in bbox]
            # Ensure xmin < xmax, ymin < ymax
            if bbox_int[0] >= bbox_int[2] or bbox_int[1] >= bbox_int[3]:
                continue
            cleaned.append({
                "description": elem.get("description", ""),
                "bbox": bbox_int,
            })

        if cleaned:
            return cleaned

    print("  [ERROR] None of the JSON candidates contained valid elements.")
    return None


def model_run_dir(model_name: str, output_dir: Optional[Path] = None) -> Path:
    """Resolve an isolated output directory for one model run."""
    if output_dir is not None:
        return Path(output_dir)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._")
    if not safe_name:
        raise ValueError(f"model name cannot form a safe run directory: {model_name!r}")
    return BASE_OUTPUT_DIR / "runs" / f"{safe_name}__{PROMPT_VERSION}"


def save_json_atomic(path: Path, payload) -> None:
    """Write experiment state atomically so an interrupted run keeps prior progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def draw_bboxes_on_image(
    img: Image.Image,
    elements: List[Dict],
    orig_size: Tuple[int, int],
    save_path: Path,
    show: bool = False,
):
    """
    Draw normalised [0,1000] bboxes onto the original-resolution image and save.
    """
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    orig_w, orig_h = orig_size

    # Try to load a truetype font; fall back to PIL default
    font = None
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=18)
            except Exception:
                continue
            break

    colors = [
        "#00FF00", "#FF4444", "#4488FF", "#FFAA00", "#FF00FF",
        "#00FFFF", "#FFFF00", "#AA44FF", "#44FFAA", "#FF8844",
    ]

    for i, elem in enumerate(elements):
        color = colors[i % len(colors)]
        xmin, ymin, xmax, ymax = elem["bbox"]

        # Map normalised [0,1000] → physical pixels
        px_xmin = int(xmin / 1000.0 * orig_w)
        px_ymin = int(ymin / 1000.0 * orig_h)
        px_xmax = int(xmax / 1000.0 * orig_w)
        px_ymax = int(ymax / 1000.0 * orig_h)

        # Draw rectangle
        draw.rectangle([px_xmin, px_ymin, px_xmax, px_ymax], outline=color, width=4)

        # Draw label
        label = f"#{i + 1} {elem['description'][:60]}"
        # Background for text
        if font:
            tb = draw.textbbox((px_xmin, px_ymin - 22), label, font=font)
        else:
            tb = draw.textbbox((px_xmin, px_ymin - 14), label)
        text_h = tb[3] - tb[1]
        draw.rectangle(
            [px_xmin, px_ymin - text_h - 8, px_xmin + (tb[2] - tb[0]) + 8, px_ymin],
            fill=color,
        )
        draw.text(
            (px_xmin + 4, px_ymin - text_h - 4),
            label,
            fill="white",
            font=font,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    draw_img.save(str(save_path), quality=95)
    print(f"  [SAVED] Annotated image -> {save_path}")
    if show:
        draw_img.show()


# ============================================================================
#  Main test loop
# ============================================================================

def run_rig_test(
    indices: Optional[List[int]] = None,
    inst_categories: Optional[List[int]] = None,
    n_samples: int = 5,
    show_images: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Optional[Path] = None,
    overwrite_existing: bool = False,
    resume: bool = False,
):
    """
    Run RIG grounding test on selected CharXiv samples.

    Parameters
    ----------
    indices : list[int] | None
        Specific question indices to test (0-based). Overrides n_samples.
    inst_categories : list[int] | None
        Filter to specific instruction categories (1-4).
    n_samples : int
        Number of samples to run when indices is None.
    show_images : bool
        Whether to display the annotated image.
    """
    output_dir = model_run_dir(model_name, output_dir)
    if resume and overwrite_existing:
        raise ValueError("resume and overwrite_existing cannot be enabled together")
    directory_nonempty = output_dir.exists() and any(output_dir.iterdir())
    protected_snapshot = (output_dir / "RUN_INFO.md").exists()
    if protected_snapshot and not overwrite_existing:
        raise FileExistsError(
            f"Protected experiment snapshot: {output_dir}. Choose another output directory."
        )
    if directory_nonempty and not overwrite_existing and not resume:
        raise FileExistsError(
            f"Run directory is not empty: {output_dir}. Choose another --output-dir, "
            "pass --resume, or pass --overwrite-existing explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "rig_results.json"
    status_path = output_dir / "run_status.json"

    all_results = []
    run_status = None
    if resume:
        if results_path.exists():
            loaded_results = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_results, list):
                raise ValueError(f"resume file must contain a result list: {results_path}")
            all_results = loaded_results
        if status_path.exists():
            run_status = json.loads(status_path.read_text(encoding="utf-8"))
        existing_models = {str(row.get("model_name", model_name)) for row in all_results}
        if existing_models and existing_models != {model_name}:
            raise ValueError(
                f"cannot resume {model_name!r}; existing results use models {sorted(existing_models)}"
            )
        existing_prompt_versions = {
            str(row.get("prompt_version", "legacy-v1")) for row in all_results
        }
        status_prompt_version = (
            str(run_status.get("prompt_version", "legacy-v1"))
            if run_status is not None else PROMPT_VERSION
        )
        if (
            (existing_prompt_versions and existing_prompt_versions != {PROMPT_VERSION})
            or status_prompt_version != PROMPT_VERSION
        ):
            raise ValueError(
                "cannot resume across prompt versions; choose a new --output-dir. "
                f"current={PROMPT_VERSION!r}, existing_results={sorted(existing_prompt_versions)}, "
                f"existing_status={status_prompt_version!r}"
            )

    df = load_charxiv_reasoning(TSV_PATH)

    # --- Select samples ---
    if indices is not None:
        subset = df[df["index"].isin(indices)]
    elif inst_categories is not None:
        # Sample n_samples *per category* so each category gets equal coverage
        frames = []
        for cat in inst_categories:
            cat_df = df[df["inst_category"] == cat]
            k = min(n_samples, len(cat_df))
            if k > 0:
                frames.append(cat_df.sample(k, random_state=42))
        subset = pd.concat(frames, ignore_index=True) if frames else df.head(0)
    else:
        subset = df.sample(min(n_samples, len(df)), random_state=42)

    print(f"\nRunning RIG test on {len(subset)} sample(s)\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")

    requested_indices = [int(v) for v in subset["index"].tolist()]
    completed_indices = {int(row["index"]) for row in all_results}
    if run_status is None:
        run_status = {
            "model_name": model_name,
            "prompt_version": PROMPT_VERSION,
            "guided_json": True,
            "question_source_policy": "prefer_raw_question_then_question",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "requested_indices": requested_indices,
            "completed_indices": sorted(completed_indices),
            "failures": [],
        }
    else:
        run_status["resumed_at"] = datetime.now(timezone.utc).isoformat()
        run_status["requested_indices"] = sorted(
            set(int(v) for v in run_status.get("requested_indices", [])) | set(requested_indices)
        )
        run_status["completed_indices"] = sorted(completed_indices)
    save_json_atomic(results_path, all_results)
    save_json_atomic(status_path, run_status)

    if completed_indices:
        print(f"Resume: skipping {len(set(requested_indices) & completed_indices)} completed sample(s)")
        subset = subset[~subset["index"].astype(int).isin(completed_indices)]

    for _, row in subset.iterrows():
        idx = int(row["index"])
        question, question_field = get_rig_question(row)
        inst_cat = int(row["inst_category"])
        answer = str(row.get("answer", ""))

        cat_names = {1: "Text-in-Chart", 2: "Text-in-General",
                     3: "Number-in-Chart", 4: "Number-in-General"}
        cat_name = cat_names.get(inst_cat, f"Unknown({inst_cat})")

        print(f"\n--- [{cat_name}] Q #{idx} ---")
        print(f"  Question: {question[:120]}...")
        print(f"  GT Answer: {answer[:80]}")

        # Load image
        try:
            img = get_image(row)
        except Exception as e:
            print(f"  [SKIP] Cannot load image: {e}")
            run_status["failures"].append({"index": idx, "stage": "image", "error": str(e)})
            save_json_atomic(status_path, run_status)
            continue
        orig_size = img.size

        # Encode
        data_uri, _ = encode_image_to_base64(img)

        # Build prompt
        prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)

        # Call model
        t0 = time.time()
        try:
            extra_body = build_rig_extra_body()
            
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=MAX_TOKENS,
                messages=build_rig_messages(prompt, data_uri),
                extra_body=extra_body,
            )
        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            run_status["failures"].append({"index": idx, "stage": "api", "error": str(e)})
            save_json_atomic(status_path, run_status)
            continue

        elapsed = time.time() - t0
        msg = response.choices[0].message

        # Qwen-style models may put reasoning in a separate field
        reasoning = getattr(msg, "reasoning_content", None)
        raw_output = msg.content or ""

        # If the model used a reasoning field, the real answer is in content;
        # otherwise everything (including thinking) is in content.
        if reasoning:
            print(f"  (reasoning_content: {len(reasoning)} chars, "
                  f"content: {len(raw_output)} chars)")

        # Token usage
        if hasattr(response, "usage") and response.usage:
            print(f"  Tokens: prompt={response.usage.prompt_tokens}, "
                  f"completion={response.usage.completion_tokens}, "
                  f"total={response.usage.total_tokens}  ({elapsed:.1f}s)")

        print(f"  Raw output (first 300 chars): {raw_output[:300]}...")

        # Parse JSON
        elements = parse_rig_response(raw_output)

        if elements is None or len(elements) == 0:
            print("  [FAIL] No valid elements extracted.")
            # Save raw output for debugging
            debug_path = output_dir / f"q{idx:04d}_raw_fail.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw_output, encoding="utf-8")
            run_status["failures"].append({
                "index": idx,
                "stage": "parse",
                "raw_output_path": str(debug_path),
                "reasoning_content": reasoning,
            })
            save_json_atomic(status_path, run_status)
            continue

        print(f"  [OK] Extracted {len(elements)} element(s):")
        for j, elem in enumerate(elements):
            print(f"    #{j + 1}: {elem['description'][:80]}  "
                  f"bbox={elem['bbox']}")

        # Draw and save
        save_name = f"q{idx:04d}_cat{inst_cat}_rig.jpg"
        save_path = output_dir / save_name
        draw_bboxes_on_image(img, elements, orig_size, save_path, show=show_images)

        # Accumulate result
        result_entry = {
            "index": idx,
            "model_name": model_name,
            "prompt_version": PROMPT_VERSION,
            "guided_json": True,
            "inst_category": inst_cat,
            "category_name": cat_name,
            "question": question,
            "question_source_field": question_field,
            "dataset_question": str(row.get("question", "")),
            "gt_answer": answer,
            "model_raw_output": raw_output,
            "reasoning_content": reasoning,
            "elements": elements,
            "token_usage": {
                "prompt": response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else None,
                "completion": response.usage.completion_tokens if hasattr(response, "usage") and response.usage else None,
            },
        }
        all_results.append(result_entry)
        run_status["completed_indices"] = sorted(
            set(int(v) for v in run_status.get("completed_indices", [])) | {idx}
        )
        save_json_atomic(results_path, all_results)
        save_json_atomic(status_path, run_status)

    # ---- Save combined results ----
    run_status["finished_at"] = datetime.now(timezone.utc).isoformat()
    successful_requested = len(set(requested_indices) & {int(row["index"]) for row in all_results})
    run_status["successful_samples"] = successful_requested
    run_status["failed_samples"] = len(requested_indices) - successful_requested
    save_json_atomic(results_path, all_results)
    save_json_atomic(status_path, run_status)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {results_path}")
    print(f"Total successful samples: {successful_requested} / {len(requested_indices)}")

    return all_results


# ============================================================================
#  Quick single-question test (original-style, kept for compatibility)
# ============================================================================

def quick_single_test(
    image_path: str,
    question: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Optional[Path] = None,
    overwrite_existing: bool = False,
):
    """
    Run RIG on a single arbitrary image + question.
    Useful for ad-hoc debugging outside the CharXiv dataset.
    """
    output_dir = model_run_dir(model_name, output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_existing:
        raise FileExistsError(
            f"Run directory is not empty: {output_dir}. Choose another --output-dir, "
            "or pass --overwrite-existing explicitly."
        )
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    data_uri, _ = encode_image_to_base64(img)
    prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)

    extra_body = build_rig_extra_body()

    response = client.chat.completions.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        messages=build_rig_messages(prompt, data_uri),
        extra_body=extra_body,
    )

    raw = response.choices[0].message.content or ""
    print("=== RAW OUTPUT ===")
    print(raw)

    elements = parse_rig_response(raw)
    if elements:
        print(f"\n=== EXTRACTED {len(elements)} ELEMENTS ===")
        for j, elem in enumerate(elements):
            print(f"  #{j + 1}: {elem['description']}  bbox={elem['bbox']}")

        out_path = output_dir / "single_test_result.jpg"
        draw_bboxes_on_image(img, elements, orig_size, out_path, show=True)
    else:
        print("Failed to parse elements.")

    return elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RIG predictions on CharXiv reasoning samples.")
    parser.add_argument(
        "--formal-annotations",
        type=Path,
        help="Run exactly the image indices in annotation_1.json (annotation_0 is never loaded implicitly).",
    )
    parser.add_argument("--indices", type=int, nargs="+", help="Run explicit CharXiv indices.")
    parser.add_argument("--n-samples", type=int, default=5, help="Random sample count when no indices are given.")
    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="API model alias.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Isolated run directory; defaults to test_case/RIG_Charxiv/runs/<model-name>.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow writing into a non-empty run directory. Omit this for protected experiments.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partial run and skip indices already present in rig_results.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and paths without API calls.")
    parser.add_argument("--quick-image", help="Run one arbitrary image instead of the dataset loop.")
    parser.add_argument("--quick-question", default="Provide the values on the blue curve.")
    args = parser.parse_args()

    if args.quick_image:
        quick_single_test(
            args.quick_image,
            args.quick_question,
            model_name=args.model_name,
            output_dir=args.output_dir,
            overwrite_existing=args.overwrite_existing,
        )
    else:
        selected_indices = args.indices
        if args.formal_annotations:
            annotation_data = json.loads(args.formal_annotations.read_text(encoding="utf-8"))
            selected_indices = sorted({
                int(Path(str(item["path"]).replace("\\", "/")).stem)
                for item in annotation_data.get("images", [])
            })
            if len(selected_indices) != 41:
                raise SystemExit(
                    f"Expected 41 formal annotation samples, found {len(selected_indices)} "
                    f"in {args.formal_annotations}"
                )
        if args.dry_run:
            resolved_output = model_run_dir(args.model_name, args.output_dir)
            print(f"Model: {args.model_name}")
            print(f"Output directory: {resolved_output}")
            print(f"Selected samples: {len(selected_indices) if selected_indices is not None else args.n_samples}")
            print(f"Directory exists: {resolved_output.exists()}")
            print(f"Protected snapshot: {(resolved_output / 'RUN_INFO.md').exists()}")
            print(
                "Would refuse non-empty directory: "
                f"{((resolved_output / 'RUN_INFO.md').exists() and not args.overwrite_existing) or (resolved_output.exists() and any(resolved_output.iterdir()) and not args.overwrite_existing and not args.resume)}"
            )
            raise SystemExit(0)
        run_rig_test(
            indices=selected_indices,
            n_samples=args.n_samples,
            show_images=args.show_images,
            model_name=args.model_name,
            output_dir=args.output_dir,
            overwrite_existing=args.overwrite_existing,
            resume=args.resume,
        )
