import os
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

# 数据集路径
tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"

# 输出目录
output_dir = r"C:\Users\19639\Desktop\sunset of ocrreasoning\images"
os.makedirs(output_dir, exist_ok=True)

try:
    # 读取 TSV 文件
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"成功加载数据集，共包含 {len(df)} 条数据。\n")
    print("可用列名:", df.columns.tolist())
    print("-" * 50)

    # 取前 10 条数据
    subset = df.head(20)
    saved_count = 0

    for index, row in subset.iterrows():
        item_id = row.get('index', index)
        question = str(row.get('question', ''))[:80]
        print(f"[{index}] ID={item_id} | {question}...")

        img_str = row.get('image', None)
        if pd.notna(img_str):
            try:
                img_data = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_data))

                filename = f"ocr_reasoning_{int(item_id):03d}.png"
                save_path = os.path.join(output_dir, filename)
                img.save(save_path, "PNG")
                print(f"  → 已保存: {save_path} ({img.size[0]}×{img.size[1]}, 原尺寸无损)")
                saved_count += 1
            except Exception as e:
                print(f"  → [图片解析报错]: {e}")
        else:
            print(f"  → [图片状态]: 该条目没有图片数据")

        print("-" * 50)

    print(f"\n完成: 共保存 {saved_count}/10 张图片到 {output_dir}")

except FileNotFoundError:
    print(f"未找到文件: {tsv_path}。请检查是否已正确生成或放置了该数据集文件。")