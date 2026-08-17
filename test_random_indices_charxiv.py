import random
import json

# 设置随机种子，确保结果可复现
random.seed(813)

# 从0-999（共1000个实例）中随机抽取40个不重复的整数
indices = sorted(random.sample(range(1000), 40))

# 保存为JSON文件
output_path = "test_case/random_indices_charxiv.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(indices, f, ensure_ascii=False)

print(f"已生成 {len(indices)} 个随机索引，保存至 {output_path}")
print(f"范围: {indices[0]} ~ {indices[-1]}")