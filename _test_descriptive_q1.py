"""One-off: test RIG on CharXiv descriptive_val Q#1."""
import sys
sys.path.insert(0, ".")
from test_RIG_on_Charxiv import *
import pandas as pd

df = pd.read_csv(r"C:\Users\19639\LMUData\CharXiv_reasoning_val.tsv", sep="\t")
row = df[df["index"] == 3].iloc[0]

img = Image.open(r"C:\Users\19639\LMUData\images\CharXiv_reasoning_val\1.jpg").convert("RGB")
orig_size = img.size
data_uri, _ = encode_image_to_base64(img)
question = str(row["question"])
answer = str(row["answer"])

print(f"Q: {question[:150]}")
print(f"A: {answer[:80]}")


RIG_USER_PROMPT_TEMPLATE = """Task: Grounding target element.

__QUESTION_PLACEHOLDER__

RULES
1.Element needs:
   - "description": short label (< 20 words) saying what the element is and why it matters
   - "bbox": [xmin, ymin, xmax, ymax] — integers, normalised [0, 1000], (0,0)=top-left
2. **TIGHT boxes only.** The bbox must tightly enclose the element itself (text line, legend key,
   tick label, data point, axis title, etc.). Do NOT draw a box around an entire curve or an
   entire chart region — box INDIVIDUAL text labels, tick marks, data markers, and legend entries.

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
prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)

extra_body = {
    "mm_processor_kwargs": {"images_kwargs": {"size": {
        "longest_edge": 8192 * 32 * 32, "shortest_edge": 4096 * 32 * 32}}}
}

resp = client.chat.completions.create(
    model=MODEL_NAME, max_tokens=MAX_TOKENS,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
    extra_body=extra_body,
)

raw = resp.choices[0].message.content or ""
print(f"\n--- RAW OUTPUT ---\n{raw[:600]}")

elements = parse_rig_response(raw)
if elements:
    for j, e in enumerate(elements):
        print(f"  #{j+1}: {e['description'][:80]}  bbox={e['bbox']}")
    draw_bboxes_on_image(img, elements, orig_size,
                         Path("test_case/RIG_Charxiv/descriptive_q1.jpg"), show=True)
else:
    print("Parse failed")
