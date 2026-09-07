# CharXiv GT Mask-In 实验

## 实验目的

GT Mask-In 是一个 oracle 干预实验，用于回答：如果模型只接收人工标注的必要视觉证据，保留证据在原图中的空间位置，能否保持或提高下游问答准确率？

它验证的是“人工必要证据是否具有功能充分性，以及去除无关视觉内容是否有帮助”。它本身不能单独证明某一种 learned re-embedding 模块有效；后续还需要 Full Image、同面积 Random Mask-In、GT Mask-Out 和 learned/predicted evidence 条件作为对照。

## 变换定义

对原图 $I$ 和该题的 GT 框集合 $B=\{b_i\}_{i=1}^{n}$，先在原图分辨率上构造并集掩码：

$$
M(p)=\mathbb{1}\left[p\in\bigcup_{i=1}^{n} b_i\right].
$$

GT Mask-In 图像为：

$$
I_{\mathrm{in}}(p)=M(p)I(p)+(1-M(p))c,
$$

其中 $c$ 默认为中性灰色 `(127,127,127)`。脚本不会裁剪、拼接、改变画布尺寸、绘制 bbox 边框，且不会把 annotation description 提供给模型。重叠或完全相同的 bbox 自动按像素并集合并；语义线索仍在结果元数据中分别保留。

## 脚本与默认数据

脚本为 `test_GT_MaskIn_on_Charxiv.py`。默认读取：

- `test_case/RIG_Charxiv/annotation_1.json`：41 题、303 个必要线索；不读取 `annotation_0.json`。
- `C:/Users/19639/LMUData/CharXiv_reasoning_val.tsv`。
- `C:/Users/19639/LMUData/images/CharXiv_reasoning_val/`。

默认使用 TSV 的 `question` 字段，而不是 `raw_question`，因为标准 CharXiv full-image 基线也是用 `question`，这样答案准确率才可直接比较。

## 推荐运行流程

先验证 41 题、标注和图片路径，不写文件也不调用 API：

```powershell
python test_GT_MaskIn_on_Charxiv.py --model-name Qwen3.5-9B --dry-run
```

只生成全部 GT Mask-In 输入和二值 mask，供人工检查：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --prepare-only `
  --output-dir test_case/RIG_Charxiv/gt_mask_in/prepared-v1
```

先做单题 smoke test：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --indices 1 `
  --output-dir test_case/RIG_Charxiv/gt_mask_in/smoke/Qwen3.5-9B
```

运行完整 41 题：

```powershell
python test_GT_MaskIn_on_Charxiv.py --model-name Qwen3.5-9B
python test_GT_MaskIn_on_Charxiv.py --model-name Qwen3.5-27B
```

默认输出目录按模型隔离：

```text
test_case/RIG_Charxiv/gt_mask_in/
  Qwen3.5-9B__gt-mask-in-v1/
  Qwen3.5-27B__gt-mask-in-v1/
```

截至 2026-09-01，上述两个正式目录均已生成 `FROZEN.json`：9B清单记录95个文件，27B清单记录93个文件。runner会拒绝对带冻结标记的目录执行推理、resume、重建或重新评测；新实验必须使用新的condition默认目录或显式指定新的`--output-dir`。可随时检查冻结文件是否改变：

```powershell
python test_GT_MaskIn_on_Charxiv.py --model-name Qwen3.5-9B --verify-frozen
python test_GT_MaskIn_on_Charxiv.py --model-name Qwen3.5-27B --verify-frozen
```

## 统一Evidence Condition Runner

同一脚本现支持：

- `gt-mask-in`：只保留GT联合区域；历史正式结果使用该condition。
- `full-image`：原图单图输入，用于同runner基线。
- `gt-mask-out`：只将GT联合区域替换为固定灰色，其余像素不变。
- `random-mask-in`：对GT联合mask做确定性循环平移，在128个候选中选择与GT重叠最小者；精确保留联合mask的像素面积与形状。不同重复实验用不同`--random-seed`。
- `gt-roi`：先输入一张GT Mask-In布局图，再按annotation文件顺序输入每个GT bbox的原始ROI裁剪，最后输入问题。它不增加annotation description或框外像素。

除`gt-mask-in`历史目录外，新condition默认写入：

```text
test_case/RIG_Charxiv/evidence_conditions/<condition>/<model>__evidence-conditions-v1/
```

例如只准备5题并执行像素与输入顺序校验：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --condition gt-roi `
  --indices 1 17 8 244 346 `
  --prepare-only `
  --output-dir test_case/RIG_Charxiv/evidence_conditions/smoke-5/gt-roi

python test_GT_MaskIn_on_Charxiv.py `
  --condition gt-roi `
  --output-dir test_case/RIG_Charxiv/evidence_conditions/smoke-5/gt-roi `
  --validate-prepared
```

`model_input_manifest.json`不保存Base64，而是明确记录模型输入图片的路径、顺序、`detail=high`和最后的question文本。`preparation_validation_report.json`逐题验证文件存在性、图片可解码性、mask像素方向、随机mask等面积、ROI数量/像素来源以及“全部图片在前、问题在后”的消息结构。

502 或进程中断后，使用完全相同的配置继续，只跳过成功样本：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --resume
```

如果服务端口改变，显式指定 OpenAI-compatible base URL：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --api-base http://HOST:PORT/v1
```

完整成功后，可以调用项目已有的 CharXiv judge：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --resume `
  --evaluate `
  --judge-model qwen-plus `
  --judge-nproc 4 `
  --judge-timeout 120 `
  --judge-retry 3
```

脚本默认拒绝评测不完整运行，以免把 API 失败误算成模型错误。`judge-timeout` 与 `judge-retry` 限制外部 judge 故障时的等待时间；它们不改变模型原始预测。

### 已有运行的无损重建

早期 `gt-mask-in-v1` 运行把完整 thinking 与最终答案一起写进了 Excel。Excel 单元格最多容纳32767字符，因此长输出会产生截断警告。使用下面的命令可以直接从完整的 `predictions.json` 重建评测输入，不会调用模型，也不会覆盖原始 `predictions.json` 或 `predictions.xlsx`：

```powershell
python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-9B `
  --rebuild-existing

python test_GT_MaskIn_on_Charxiv.py `
  --model-name Qwen3.5-27B `
  --rebuild-existing
```

重建后新增：

- `predictions_normalized.json`：保留完整 `model_raw_output`，并增加 `judge_prediction` 与 `answer_status`。
- `predictions_for_judge.xlsx`：只在 `prediction` 列放置最终回答，不写入超长 thinking。
- `answer_quality_report.json`：分别统计 HTTP 完成、有效答案、token 截断、空答案和请求失败。

`finish_reason=length` 即使已经出现 `</think>`，仍统一视为未在固定 token 预算内给出完整答案；judge 输入使用明确的 `NO VALID ANSWER` 标记。后续 `--evaluate` 会优先读取 `predictions_for_judge.xlsx`。

CharXiv judge 现已支持从部分 `.pkl` 缓存断点续评：pending keys 与 pending tasks 使用同一组行索引构造，不再因已有缓存触发 `len(keys) != len(tasks)`。因此遇到中断后可直接重复原评测命令，无需删除缓存。

评测入口会优先选择 `predictions_for_judge.xlsx`，只有该文件不存在时才回退到旧的 `predictions.xlsx`。`auxeval` 也会把Excel中的空值/`NaN`安全转换为空字符串，确保无答案样本计0而不会以`TypeError`中止整组并发评测。启动时控制台会打印实际使用的 `Evaluation workbook`，正式评测应确认其文件名为 `predictions_for_judge.xlsx`。

## 输出文件

- `experiment_config.json`：模型、数据哈希、mask 参数、问题字段和生成参数。
- `run_status.json`：请求、成功、失败 index 和完整性状态。
- `predictions.json`：逐题 GT、mask 统计、原始答案、reasoning、token 和耗时。
- `predictions.xlsx`：保留 CharXiv 的 `grading_query`，可直接交给现有 judge。
- `predictions_for_judge.xlsx`：对已有长输出执行 `--rebuild-existing` 后生成的规范评测文件。
- `masked_images/*.png`：模型实际接收的图像。
- `binary_masks/*.png`：对应的严格二值并集掩码。

其中 `visible_ratio` 是 GT 并集面积占整图面积的比例；`summed_box_ratio` 会重复计算重叠区域；二者之差记录在 `overlap_pixels` 中。

## 公平对比要求

GT Mask-In 和 Full Image 必须固定相同的模型 checkpoint/API、`question` 字段、temperature、max tokens、thinking 设置、图片分辨率策略和答案 judge。正式结论至少报告：

- Full Image Accuracy；
- GT Mask-In Accuracy；
- `GT Mask-In − Full Image` 的逐题配对差值；
- bootstrap 95% 置信区间或配对置换检验；
- 按 `inst_category`、GT 框数量和 `visible_ratio` 的分组结果。

建议再运行 `bbox_padding = 0, 5, 10` 的敏感性分析。若只有大 padding 才能恢复性能，说明当前 bbox 可能截断上下文，不能直接把提升归因于 re-embedding。
