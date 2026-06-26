import random
import json

# 设置随机种子，确保结果可复现
random.seed(624)

# 从0-1068（共1069个实例）中随机抽取200个不重复的整数
indices = sorted(random.sample(range(1069), 200))

# 保存为JSON文件
output_path = "test_case/random_indices.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(indices, f, ensure_ascii=False)

print(f"已生成 {len(indices)} 个随机索引，保存至 {output_path}")
print(f"范围: {indices[0]} ~ {indices[-1]}")



##################################################
# 读取JSON文件
import json

with open('test_case/random_indices.json', 'r') as f:
    indices = json.load(f)

    # 从完整数据集中提取子集
subset = dataset.iloc[indices]