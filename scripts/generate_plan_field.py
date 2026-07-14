"""
为所有推理轨迹 JSON 添加 'plan' 字段 —— 任务导向的步骤描述，
不包含 GT 数值、中间结果、推理结论。

核心原则：plan 只描述 **做什么**，不描述 **是什么**。
"""
import json
import os
import re
import glob

TRACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_case', 'reasoning_traces')


# ═══════════════════════════════════════════════════════════════════
#  Read step: 剥除具体数值，保留语义目标
# ═══════════════════════════════════════════════════════════════════
def clean_read_description(desc, expected_value):
    ev = str(expected_value).strip() if expected_value else ''
    d = desc.strip()

    # --- 模板匹配（英文）—— 先于 ev 剥离，保证模板能匹配原始文本 ---
    # 注意：更具体的模式必须放在更泛化的模式之前！
    templates_en = [
        # "the number 150 in the phrase '150 calories' in reasoning step 1"
        # → strip the quoted phrase to avoid leaking value
        (r"^the\s+number\s+[\d.]+\s+in\s+the\s+phrase\s+'[^']*'\s+in\s+(.+)$",
         lambda m: f'Read the numeric value mentioned in {m.group(1)}'),

        # "the number 10.1 for 2023 total revenue" → "Read the 2023 total revenue value"
        (r'^the\s+number\s+[\d.]+\s+for\s+(.+)$',
         lambda m: f'Read the {m.group(1)} value from the chart'),

        # "the number 0.4 for the black bar in 2000" → "Read the black bar value for 2000"
        (r'^the\s+number\s+[\d.]+\s+for\s+(the\s+.+)$',
         lambda m: f'Read {m.group(1)} for the specified year'),

        # "the percentage 7% for Brand 2 in 2023" → "Read the Brand 2 percentage for 2023"
        (r'^the\s+percentage\s+[\d.%]+\s+for\s+(.+)$',
         lambda m: f'Read the {m.group(1)} percentage from the chart'),

        # "the price 399元 for X" → "Read the price of X"
        (r'^the\s+price\s+[\d.]+元?\S*\s+for\s+(.+)$',
         lambda m: f'Read the price of {m.group(1)}'),

        # "the price text for X" → "Read the price of X"
        (r'^the\s+price\s+text\s+for\s+(.+)$',
         lambda m: f'Read the price of {m.group(1)}'),

        # "the number 7 in the first column" → "Read the value in the first column"
        (r'^the\s+number\s+[\d.]+\s+in\s+(.+)$',
         lambda m: f'Read the value in {m.group(1)}'),

        # "the number 0.7 for the black bar in 2007" (already handled above, fallback)
        (r'^the\s+number\s+[\d.]+\s+for\s+(the\s+.+in\s+\d{4})$',
         lambda m: f'Read the {m.group(1)} value'),

        # "reading the number X for Y"
        (r'^reading\s+the\s+number\s+[\d.]+\s+for\s+(.+)$',
         lambda m: f'Read the {m.group(1)} value from the chart'),

        # "reading the value represented by X" → keep
        (r'^reading\s+the\s+value\s+represented\s+by\s+(.+)',
         lambda m: f'Read the value represented by {m.group(1)}'),

        # "reading the percentage X for Y"
        (r'^reading\s+the\s+percentage\s+[\d.%]+\s+for\s+(.+)$',
         lambda m: f'Read the {m.group(1)} percentage from the chart'),

        # "reading the height value X from the image" → "Read the height measurement"
        (r'^reading\s+the\s+(\w+)\s+value\s+[\d.]+\s+from\s+the\s+image$',
         lambda m: f'Read the {m.group(1)} measurement from the image'),

        # "reading the width value X from the image"
        (r'^reading\s+the\s+(\w+)\s+value\s+[\d.]+\s+from\s+the\s+image$',
         lambda m: f'Read the {m.group(1)} measurement from the image'),

        # "reading the total percentage value from the chart scale" → keep
        (r'^reading\s+the\s+(.+)\s+from\s+the\s+chart', None),

        # "reading the percentage of adults who contacted..." → keep
        (r'^reading\s+the\s+percentage\s+of\s+(.+)', None),
    ]

    for pattern, replacer in templates_en:
        m = re.match(pattern, d, re.IGNORECASE)
        if m:
            if replacer is None:
                return d  # keep as-is
            return replacer(m)

    # --- 如果没匹配到任何模板，使用通用清洗作为 fallback ---
    # 先移除 expected_value
    if ev and ev in d:
        d = re.sub(r'\s*' + re.escape(ev) + r'\s*', ' ', d)
        d = re.sub(r'\s+', ' ', d).strip()

    # 移除 "the number", "the price" 等泄露值的引导词
    d = re.sub(r'^the\s+(number|price|percentage|value|text)\s+', '', d, flags=re.IGNORECASE)
    d = re.sub(r'^for\s+', '', d)
    d = re.sub(r'^:\s*', '', d)
    d = re.sub(r"in\s+the\s+phrase\s+'[^']*'\s+", '', d)  # 移除 '150 calories' 等引用
    d = d.strip().rstrip('.')

    # 中文 → 如果干净就保留
    if re.search(r'[一-鿿]', d):
        if re.match(r'^(读取|获取|找到|识别|确定|查看)', d):
            return d
        return '读取' + d if not d.startswith('读取') else d

    # 英文 → 加 Read 前缀
    if d and not re.match(r'^(Read|Locate|Find|Identify|Determine)', d, re.IGNORECASE):
        d = 'Read ' + d[0].lower() + d[1:]
    elif not d:
        d = desc.strip()  # fallback 回原文
    return d


# ═══════════════════════════════════════════════════════════════════
#  Compute step: 剥除表达式/结果，保留操作语义
# ═══════════════════════════════════════════════════════════════════
def clean_compute_description(desc):
    d = desc.strip()

    # 纯表达式： "2 + 4 + 4 = 10" → "Sum the values"
    if re.match(r'^[\d.+\-×÷*/x()\s=]+$', d):
        if '+' in d:
            return 'Sum the given values'
        if '-' in d:
            return 'Subtract the given values'
        if re.search(r'[×*]', d):
            return 'Multiply the given values'
        if re.search(r'[÷/]', d):
            return 'Divide the given values'
        return 'Calculate the result'

    # "10 + x = 14" → "Solve for the unknown value"
    if re.match(r'^[\d.+\-×÷*/x()\s=]+$', d.replace('x', '0')):
        return 'Solve for the unknown value'

    # "x = 14 - 10" → "Calculate the unknown from the equation"
    if re.match(r'^x\s*=\s*[\d.+\-×÷*/()\s=]+$', d):
        return 'Calculate the unknown value from the equation'

    # "x = 4" → "State the final answer"
    if re.match(r'^x\s*=\s*\d+$', d):
        return 'State the final computed value'

    # "Adding these numbers we get 7+2+1+4=14" → "Sum the values"
    m = re.match(r'^(adding|summing)\s+.+?\s+(we\s+get\s+|=\s*)[\d.+\-×÷*/x()\s]+$', d, re.IGNORECASE)
    if m:
        return 'Sum the given values'

    # "sum the four values" → keep
    # "divide sum by 4 to get average" → keep
    # "calculate 2023 Brand 2 revenue" → keep
    if re.match(r'^(calculate|sum|add|subtract|multiply|divide|compute|find)', d, re.IGNORECASE):
        return d

    # 中文 compute
    if re.search(r'[一-鿿]', d):
        if re.match(r'^(计算|求和|相加|减去|乘以|除以|求|得到)', d):
            return d
        return d  # 中文通常已经很干净

    return d


# ═══════════════════════════════════════════════════════════════════
#  Logical step: 结论式 → 任务式
# ═══════════════════════════════════════════════════════════════════
def clean_logical_description(desc):
    d = desc.strip()

    # "Identify black bars represent X" → "Determine which bars represent X"
    m = re.match(r'^Identify\s+(.+?)\s+represent\b(.+)', d, re.IGNORECASE)
    if m:
        entity = m.group(1).strip()
        meaning = m.group(2).strip()
        return f'Determine which {entity} represents{meaning}'

    # "Similarly, for the second column, sum is 14" → "Apply the pattern to the second column"
    m = re.match(r'^(Similarly,?\s*)?for\s+the\s+(.+?),\s*sum\s+is\s+\d+', d, re.IGNORECASE)
    if m:
        return f'Apply the sum pattern to {m.group(2).strip()}'

    # "For the third column, sum is 14" → same handling
    m = re.match(r'^For\s+the\s+(.+?),\s*sum\s+is\s+\d+', d, re.IGNORECASE)
    if m:
        return f'Apply the sum pattern to {m.group(1).strip()}'

    # "2 + 4 + 4 + x = 14" → "Set up the equation for the column"
    if re.match(r'^[\d.+\-×÷*/x()\s=]+$', d.replace('x', '0')):
        return 'Set up the equation using the column sum pattern'

    # "X satisfies Y condition" → "Check whether X satisfies Y"
    m = re.match(r'^(.+?)\s+(satisfies|does\s+not\s+satisfy)\s+(.+)', d, re.IGNORECASE)
    if m:
        return f'Check whether {m.group(1).strip()} satisfies {m.group(3).strip()}'

    # "Determine X is the cheapest" → "Determine which item is the cheapest"
    m = re.match(r'^Determine\s+(.+?)\s+is\s+(the\s+.+)', d, re.IGNORECASE)
    if m:
        return f'Determine which item is {m.group(2).strip()}'

    # "Identify X and Y" (task-oriented) → keep
    if re.match(r'^(Identify|Determine|Compare|Sort|Select|Find|Decide|Check|Verify)\b', d, re.IGNORECASE):
        return d

    # Chinese logical cleanup
    # "根据计算结果选择选项D" → "根据计算结果选择正确选项"
    d = re.sub(r'选择选项\s*[A-D]', '选择正确选项', d)
    # "比较X和Y确定最大值" → keep (already task-oriented)

    # "Identify inventory with largest code value" → keep (already task)
    if re.match(r'^Identify\s+\w+\s+with\s+', d, re.IGNORECASE):
        return d

    # "deduce X" → "Determine X"
    d = re.sub(r'^Deduce\b', 'Determine', d, re.IGNORECASE)

    # 中文任务式 → keep
    if re.search(r'[一-鿿]', d):
        if re.match(r'^(识别|比较|排序|选择|判断|检验|确认|确定|找出)', d):
            return d
        return d  # 中文通常已经很干净

    # Fallback: add "Determine" prefix if it looks like a statement
    if d and not re.match(r'^(Determine|Check|Compare|Identify|Find|Select)', d, re.IGNORECASE):
        d = 'Determine ' + d[0].lower() + d[1:]

    return d if len(d) > 3 else desc.strip()


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def generate_plan(step):
    step_type = step.get('type', '')
    desc = step.get('description', '')

    try:
        if step_type == 'read':
            return clean_read_description(desc, step.get('expected_value', ''))
        elif step_type == 'compute':
            return clean_compute_description(desc)
        elif step_type == 'logical':
            return clean_logical_description(desc)
        else:
            return desc
    except Exception as e:
        print(f"  ERROR generating plan for step type={step_type}: {e}")
        return desc


def process_all_traces(traces_dir=TRACES_DIR):
    json_files = sorted(glob.glob(os.path.join(traces_dir, '*.json')))
    if not json_files:
        print(f"ERROR: No JSON files found in {traces_dir}")
        return

    total_steps = 0
    unchanged = 0
    warnings = 0

    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for step in data.get('steps', []):
            total_steps += 1
            old_desc = step.get('description', '')
            plan = generate_plan(step)
            step['plan'] = plan

            if plan == old_desc:
                unchanged += 1

            # 检测 read 步骤的 plan 是否仍含 expected_value（非字母的值）
            if step['type'] == 'read':
                ev = str(step.get('expected_value', '')).strip()
                # 只检查纯数字/百分比泄露；字母值（如 R, D）不算
                if ev and re.search(r'\d', ev) and ev in plan:
                    print(f"  WARNING: {os.path.basename(fpath)} step {step['step_id']}: "
                          f"numeric value '{ev}' leaked in plan: '{plan[:80]}'")
                    warnings += 1

            # 检测 compute/logical 步骤是否仍含明显的数值等式
            if step['type'] in ('compute', 'logical'):
                if re.search(r'[\d.]+\s*[+\-×÷*/]\s*[\d.]+\s*=', plan):
                    print(f"  WARNING: {os.path.basename(fpath)} step {step['step_id']}: "
                          f"equation leaked in plan: '{plan[:80]}'")
                    warnings += 1

        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessed {len(json_files)} traces, {total_steps} steps total.")
    print(f"  Unchanged: {unchanged}/{total_steps} ({unchanged/total_steps*100:.1f}%)")
    print(f"  Warnings:  {warnings}")
    print(f"  Plan fields written to all trace files.")


if __name__ == '__main__':
    process_all_traces()
