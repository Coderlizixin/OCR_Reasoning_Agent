import pandas as pd
import json


def extract_and_save_errors(file_path, output_json_path):
    # 1. 读取包含推理结果的文件（以 CSV 为例，若是 Excel 请用 pd.read_excel）
    print("正在读取数据...")
    df = pd.read_excel(file_path)

    # 2. 数据对齐与清洗 (非常关键)
    # 将两列都转换为字符串，并去除首尾可能存在的不可见空格，防止误判
    df['answer_str'] = df['answer'].astype(str).str.strip()
    df['res_str'] = df['res'].astype(str).str.strip()

    # 3. 筛选出回答错误（answer 和 res 不相等）的行
    error_df = df[df['answer_str'] != df['res_str']]

    # 4. 将这些错误行的 'index' 提取为 Python 列表
    error_indices = error_df['index'].tolist()

    print(f"提取完成！共找到 {len(error_indices)} 个错误样本。")
    print(error_indices)

    # 5. 将列表保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # indent=4 可以让生成的 json 文件在记事本里具有良好的可读性
        json.dump(error_indices, f, indent=4)

    print(f"错误集 index 列表已成功保存至: {output_json_path}")
    return error_indices


def load_error_subset(error_json_path):
    # 1. 读取保存的错误集 index 列表
    with open(error_json_path, 'r', encoding='utf-8') as f:
        error_list = json.load(f)
        print(type(error_list))


# --- 运行示例 ---
# 请替换为您实际的文件路径
input_file = "outputs/Qwen3.5-9B-API/Qwen3.5-9B-API_OCR_Reasoning_qwen-plus.xlsx"
output_file = "test_case/error_indices.json"

error_list = extract_and_save_errors(input_file, output_file)
load_error_subset(output_file)
