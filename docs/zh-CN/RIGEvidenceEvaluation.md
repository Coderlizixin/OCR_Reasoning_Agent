# RIG 必要线索评估（annotation v1）

这套评估用于验证：模型即使能够作答，也未必能显式给出人工标注的全部必要视觉线索。它直接使用现有 `annotation_1.json`，不要求扩展为 schema v2。

## 评估对象

- `annotation_1.json` 是正式数据：41 个样本、303 个必要线索。
- `annotation_0.json` 是标注工具测试用例，不进入评估。
- 每个 annotation 对象的 `id` 被视为一个独立的必要语义线索；`description` 表示线索身份，bbox 表示位置。
- 两个 bbox 即使重叠或完全相同，也不会合并为一个线索。

## 为什么分成空间和语义两层

只按 IoU 逐 GT 取最大值，会让同一个预测框同时命中多个重叠 GT，从而虚增召回。本实现先产生空间候选，再执行全局一对一匹配，所以一个预测元素最多匹配一个 GT 元素。

空间候选满足以下任一条件：

1. `IoU >= threshold`，默认分别报告 0.3 和 0.5；
2. GT 被预测框覆盖至少 70%，且预测框面积不超过 GT 的 4 倍。

第二条用于保留 IoU 偏低但定位仍合理的细长元素。报告中的 `match_at_0.3` 和 `match_at_0.5` 都采用上述复合空间规则，并同时提供 `spatial_upper_bound_recall`。它是空间上可能达到的上界，不是最终必要线索召回率。

默认语义策略是 `overlap_review`：

- 26 个无 GT 重叠的样本按一对一空间匹配自动确认；
- 15 个存在 GT 重叠的样本必须结合 GT/预测描述人工裁决；
- 尚有未裁决候选的样本标记为 pending，不会被当成 0 分或混入正式语义指标。

如果需要最保守的正式结果，可使用 `review_all` 人工裁决全部样本。`spatial` 仅用于快速计算空间上界，不宜作为论文最终结果。

## 使用流程

所有命令从 `OCR-Reasoning-main` 目录执行。

先检查正式标注集：

```powershell
python evaluate_rig_grounding.py audit `
  --output test_case/RIG_Charxiv/annotation_audit.json
```

显式指定正式标注文件，运行全部 41 个样本（该命令会调用配置的模型 API）：

```powershell
python test_RIG_on_Charxiv.py --formal-annotations `
  "../RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json"
```

## 9B/27B隔离实验目录

旧版P0实验按API模型名保存在以下独立目录：

```text
test_case/RIG_Charxiv/runs/Qwen3.5-27B/
test_case/RIG_Charxiv/runs/Qwen3.5-9B/
```

历史27B实验的45个过程文件已完整复制到 `runs/Qwen3.5-27B/`，并通过SHA-256逐文件核对。运行目录非空时脚本默认拒绝启动，因此不会意外覆盖历史实验；只有显式传入 `--overwrite-existing` 才允许复用非空目录。

以下命令运行当前prompt版本；实际输出目录还会自动附加prompt版本：

```powershell
python test_RIG_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --formal-annotations "../RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json"
```

每完成一个样本都会增量保存 `rig_results.json` 和 `run_status.json`；成功样本保留模型原始输出、独立reasoning字段、token用量和标框图片，失败样本保留原始文本及失败阶段。

若运行中断，使用完全相同的命令并增加 `--resume`；脚本会跳过已经成功写入 `rig_results.json` 的index，继续运行失败或尚未处理的题目：

```powershell
python test_RIG_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --formal-annotations "../RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json" `
  --resume
```

9B运行结束后，单独生成和填写它自己的语义裁决文件：

```powershell
python evaluate_rig_grounding.py prepare-review `
  --predictions test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/rig_results.json `
  --output test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/semantic_review.json
```

最后生成9B报告：

```powershell
python evaluate_rig_grounding.py evaluate `
  --predictions test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/rig_results.json `
  --review test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/semantic_review.json `
  --answer-scores outputs/Qwen3_5-9B-API/Qwen3_5-9B-API_CharXiv_reasoning_val_qwen-plus.xlsx `
  --output test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/evaluation_report.json
```

## Prompt v4（精简版）+ Guided JSON

当前版本为 `derivation-evidence-v4-concise-guided-json`。为减少9B模型在删除检验、最小集合优化和反事实讨论上的思考循环，prompt不再要求模型证明每个框的必要性，也不再要求寻找“最小证据集”。任务被直接限定为定位下游推理所需的可见证据。

精简后仍保留两条关键约束：不能只框最终答案文字；比较、排序、argmax/argmin题需要返回尺度或判据、每个候选的观测以及标签或图例映射。遇到歧义时应框出消除歧义所需的证据，而不是在输出中反复讨论答案。

CharXiv输入优先使用干净的 `raw_question`，不再把 `question` 字段中“final answer must be grounded to text”等答案格式指令传给RIG模型；没有 `raw_question` 的数据才回退到 `question`。

批量与单题调用均通过vLLM `guided_json` 使用同一份JSON Schema，强制输出非空 `elements` 列表、描述字符串和四个 `[0,1000]` 整数坐标。本版本没有关闭thinking，也没有修改temperature。

新prompt的默认目录包含prompt版本，避免与已有P0结果混合：

```text
test_case/RIG_Charxiv/runs/Qwen3.5-9B__derivation-evidence-v4-concise-guided-json/
test_case/RIG_Charxiv/runs/Qwen3.5-27B__derivation-evidence-v4-concise-guided-json/
```

旧目录使用 `--resume` 时会因prompt版本不一致而拒绝执行。P0与P2结果不能直接混合作为同一次实验；比较模型规模时，9B和27B必须都运行相同prompt版本。

得到 `test_case/RIG_Charxiv/rig_results.json` 后，生成重叠样本的语义裁决表：

```powershell
python evaluate_rig_grounding.py prepare-review
```

打开 `test_case/RIG_Charxiv/semantic_review.json`。逐项将 `decision` 从 `null` 改成：

- `true`：预测描述与 GT 描述表达同一个必要语义线索；
- `false`：只是空间重叠，但语义不同。

裁决项同时保留图像路径、问题、GT/预测描述和双方 bbox，方便回看原图；该文件只是评估产物，不会修改 annotation v1。

完成后生成正式报告：

```powershell
python evaluate_rig_grounding.py evaluate --answer-scores `
  outputs/Qwen3_5-9B-API/Qwen3_5-9B-API_CharXiv_reasoning_val_qwen-plus.xlsx
```

答案分数表默认按 `index` 列连接，并读取 `score` 列。评估器会检查41个正式标注index是否全部有分数；Excel中其余CharXiv样本不会进入本次汇总。即使某题bbox输出失败，只要Excel中有对应index，它仍会进入答对/答错条件统计，并以bbox召回0计入端到端指标。

若只想先检查空间上界：

```powershell
python evaluate_rig_grounding.py evaluate --semantic-policy spatial `
  --output test_case/RIG_Charxiv/spatial_upper_bound_report.json
```

## 核心指标

- `necessary_recall`：每个样本匹配到的必要语义线索数 / GT 必要线索数。
- `complete_evidence_rate`：全部 GT 必要线索都被匹配的样本比例。
- `precision`：匹配线索数 / 模型预测线索数，用来约束大量泛化框。
- `union_region_coverage`：预测框并集与 GT 框并集的交集面积 / GT 框并集面积；重叠像素只计一次。
- `pred_union_area_ratio`：预测框并集占归一化整图的比例，用来发现“用大框覆盖全图”的退化行为。
- `mean_best_iou` / `mean_matched_iou`：定位接近程度的诊断量；前者不含语义约束，后者只统计最终一对一匹配。
- `overlap_subset` / `non_overlap_subset`：分别报告重叠与非重叠样本，检查结论是否由重叠标注造成。
- `by_inst_category`：当预测文件带有 CharXiv `inst_category` 时，按四类题型分别汇总。
- `answer_conditioned`：当预测结果中含 `answer_score` 或 `score` 时，分别汇总答对和答错样本的线索召回与完整率。

最终结论应以 `necessary_recall` 和 `complete_evidence_rate` 为主。`union_region_coverage` 高但必要线索召回低，说明模型大致看到了相关区域，却没有完整、逐项地识别必要证据；这正是当前阶段希望验证的现象。
