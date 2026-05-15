import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from multipart import file_path

# 你的数据集路径
tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"

try:
    # 读取 TSV 文件
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"成功加载数据集，共包含 {len(df)} 条数据。\n")
    print("可用列名:", df.columns.tolist())
    print("-" * 50)

    file_path = "test_case/image25.jpg"

    # 这里我们只查看前 3 条数据作为测试
    for index, row in df.loc[[12]].iterrows():
        print(f"\n[题目 ID]: {row.get('index', index)}")
        print(f"[问  题]: {row.get('question', '未找到question列')}")
        print(f"[标准答案]: {row.get('answer', '未找到answer列')}")

        # 提取并显示图片
        img_str = row.get('image', None)
        if pd.notna(img_str):
            try:
                # 解码 Base64
                img_data = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_data))

                if img.mode in ("RGBA", "P"):
                    img_to_save = img.convert("RGB")
                else:
                    img_to_save = img
                img_to_save.save(file_path, "JPEG")

                # 在 Windows 上会调用你系统默认的图片查看器打开
                img_to_save.show(title=f"Question {row.get('index', index)}")
            except Exception as e:
                print(f"[图片解析报错]: {e}")
        else:
            print("[图片状态]: 该条目没有图片数据")

        print("-" * 50)

except FileNotFoundError:
    print(f"未找到文件: {tsv_path}。请检查是否已正确生成或放置了该数据集文件。")