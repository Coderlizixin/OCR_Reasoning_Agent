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
OUTPUT_DIR = Path("test_case/RIG_Charxiv")

# API  – edit these to match your deployment
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.3.112.34:22250/v1",
)

# client = OpenAI(
#     api_key="sk-Sy5R85nas3TrIuwgI3oSjg",
#     base_url="https://litellm.mybigai.ac.cn/",
# )

MODEL_NAME = "Qwen3.5-27B"
MAX_TOKENS = 8192

# ============================================================================
#  Prompt templates
# ============================================================================

RIG_USER_PROMPT_TEMPLATE = """Task: Ground every chart element essential to answering a question.

IMAGE: A chart or scientific figure.
QUESTION: __QUESTION_PLACEHOLDER__

RULES
Each element needs:
- "description": short label (< 20 words) saying what the element is and why it matters
- "bbox": [xmin, ymin, xmax, ymax] — integers, normalised [0, 1000], (0,0)=top-left

OUTPUT — ONLY the JSON object (no markdown fences, no other text):
{
  "elements": [
    {
      "description": "legend label 'Joint-CNN' for the dashed red curve",
      "bbox": [680, 140, 870, 185]
    },
    {
      "description": "x-axis tick label 'Session 9'",
      "bbox": [778, 908, 835, 942]
    },
    {
      "description": "y-axis tick label '60' on the left axis",
      "bbox": [42, 455, 88, 478]
    }
  ]
}"""


# ============================================================================
#  Helpers
# ============================================================================

def load_charxiv_reasoning(tsv_path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load the CharXiv reasoning TSV.  Handles multi-line fields correctly."""
    df = pd.read_csv(tsv_path, sep="\t", nrows=n_rows)
    print(f"Loaded {len(df)} questions from {tsv_path}")
    return df


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

    all_results = []  # list of dicts to save as JSON

    for _, row in subset.iterrows():
        idx = int(row["index"])
        question = str(row["question"])
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
            continue
        orig_size = img.size

        # Encode
        data_uri, _ = encode_image_to_base64(img)

        # Build prompt
        prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)

        # Call model
        t0 = time.time()
        try:
            extra_body = {
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "size": {
                            "longest_edge": 8192 * 32 * 32,
                            "shortest_edge": 4096 * 32 * 32,
                        }
                    }
                },
            }
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }],
                extra_body=extra_body,
            )
        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
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
            debug_path = OUTPUT_DIR / f"q{idx:04d}_raw_fail.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw_output, encoding="utf-8")
            continue

        print(f"  [OK] Extracted {len(elements)} element(s):")
        for j, elem in enumerate(elements):
            print(f"    #{j + 1}: {elem['description'][:80]}  "
                  f"bbox={elem['bbox']}")

        # Draw and save
        save_name = f"q{idx:04d}_cat{inst_cat}_rig.jpg"
        save_path = OUTPUT_DIR / save_name
        draw_bboxes_on_image(img, elements, orig_size, save_path, show=show_images)

        # Accumulate result
        result_entry = {
            "index": idx,
            "inst_category": inst_cat,
            "category_name": cat_name,
            "question": question,
            "gt_answer": answer,
            "model_raw_output": raw_output,
            "elements": elements,
            "token_usage": {
                "prompt": response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else None,
                "completion": response.usage.completion_tokens if hasattr(response, "usage") and response.usage else None,
            },
        }
        all_results.append(result_entry)

    # ---- Save combined results ----
    results_path = OUTPUT_DIR / "rig_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {results_path}")
    print(f"Total successful samples: {len(all_results)} / {len(subset)}")

    return all_results


# ============================================================================
#  Quick single-question test (original-style, kept for compatibility)
# ============================================================================

def quick_single_test(image_path: str, question: str):
    """
    Run RIG on a single arbitrary image + question.
    Useful for ad-hoc debugging outside the CharXiv dataset.
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    data_uri, _ = encode_image_to_base64(img)
    prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)

    extra_body = {
        "mm_processor_kwargs": {
            "images_kwargs": {
                "size": {
                    "longest_edge": 8192 * 32 * 32,
                    "shortest_edge": 4096 * 32 * 32,
                }
            }
        },
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
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

        out_path = OUTPUT_DIR / "single_test_result.jpg"
        draw_bboxes_on_image(img, elements, orig_size, out_path, show=True)
    else:
        print("Failed to parse elements.")

    return elements


if __name__ == "__main__":
    question = "Provide the values on the blue curve."
    image_path = r"C:/Users/19639/LMUData/images/CharXiv/28.jpg"
    quick_single_test(image_path, question)
