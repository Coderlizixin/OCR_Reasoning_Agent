# H2：Planning–Grounding 协议

## 1. 研究问题

H2 检验：在同一问题、同一图像和相同图像调用次数下，先把问题拆成可见证据需求，再逐步 grounding，是否比重复执行 single-pass grounding 更完整地发现必要视觉证据。

本阶段只研究 evidence discovery/grounding，不执行最终答案推理，不引入 GT step、GT bbox、答案反馈或 re-embedding。H2 不是 H1 的重复：H1 证明必要证据在功能上有用；H2 研究 planning 是否能改善必要证据发现。

冻结版本由 runner 常量 `H2_VERSION`、`protocol.json` 和 `FROZEN.json` 共同确定。正式运行必须通过协议哈希验证。

## 2. 四种模式

| 模式 | 定义 | 作用 |
|---|---|---|
| `single` | 只读导入既有 Direct-BBox 单次结果，并复制到 H2 目录的 `inputs/` 快照 | 单次 grounding 基线 |
| `repeated-single` | 对同一问题重复执行 Direct-BBox prompt；每题次数等于该题 predicted plan 的 step 数 | 排除“只是增加调用次数”的收益 |
| `predicted-plan` | question-only planner 生成 evidence steps；每一步独立看完整图并 grounding；最终取 bbox 语义并集 | H2 主方法 |
| `reviewed-plan` | 人工只修改 predicted plan 的 `evidence_need`，保持 step 数、顺序和 ID 不变，再执行相同 grounder | 估计 planning 质量瓶颈 |

主比较是 `predicted-plan` 对 `repeated-single`。`single` 对 `repeated-single` 反映重复采样增益；`reviewed-plan` 对 `predicted-plan` 反映计划质量上限。

## 3. 固定推理条件

- 正式数据严格等于 `annotation_1.json` 中按冻结顺序排列的 41 个 index；`annotation_0.json` 不属于实验。
- question 优先读取不含答案格式指令的 `raw_question`，缺失时才使用 `question`。
- planner 不看图像；reviewer 不看图像、答案、GT bbox、annotation description 或 grounding 输出。
- planner、step grounder 和 repeated-single 的每次 completion 上限均为 12288 token。
- thinking 保持开启；temperature 不显式修改；guided JSON 保持开启。
- bbox 坐标为 `[xmin, ymin, xmax, ymax]`，整数、千分比归一化到 `[0,1000]`。
- 每个 grounding step 最多输出 16 个原子证据元素。不同语义线索即使 bbox 重合或完全相同也分别保留；只删除“描述（忽略大小写）和 bbox 都完全相同”的重复项。

## 4. 计划生成与冻结

正式实验必须分成两个阶段，不能在 predicted grounding 运行中临时重新采样计划。

1. 运行 `predicted-plan --plan-only --formal`，生成 `plans.json`。
2. 若 HTTP/request failure 为 0，则冻结 plan-only 目录并立即执行 `--verify-frozen`。
3. `predicted-plan`、`repeated-single` 和 `reviewed-plan` 都引用同一份冻结 `plans.json` 及其 SHA-256。

固定预算内的 `truncated` 或 `parse_failed` plan 是模型能力失败，不得重生成、人工修复或从统计中删除。该题保存为空计划：predicted/repeated/reviewed 都执行 0 次图像调用并按无证据处理。正式报告以全部 41 题为主；valid-plan 子集只能作为次要诊断。

## 5. Reviewed-plan 盲法

review template 只包含问题、predicted steps、review 状态和备注。人工可以修改 `evidence_need`，但必须保持：

- step 总数不变；
- step 顺序不变；
- 每个 `step_id` 不变；
- 不加入可见数值、候选答案、最终答案或 GT 信息；
- predicted plan 失败的题保持 `source_invalid` 和空 steps。

review 文件出现未授权字段时 runner 直接拒绝加载。正式 review 完成后必须记录 reviewer identity；review 内容及 SHA-256 会保存在 H2 运行配置和最终冻结目录中。

## 6. 状态与失败计入

- HTTP/API 异常经过固定重试仍失败，记为 `request_failed`。存在 request failure 的目录不能冻结，应使用 `--resume` 修复基础设施失败。
- `finish_reason=length`、无闭合 JSON 或 JSON 无有效 elements 是固定预算下的模型失败，不通过增加 token 或重采样补跑。
- 某题部分 step 失败但其他 step 有 bbox 时，题目可以是 sample-level `valid`，但必须单独报告 plan-level、step-level 和 sample-level failure rate。
- 所有模式保存逐 step 状态、raw output、token、wall-clock latency、bbox、并集和 provenance。多步结果不能只报告 grounding 指标而隐藏计算开销。

## 7. 指标与预注册结论

评估沿用 `evaluate_rig_grounding.py` 的一对一匹配。主设置为 IoU/受控包含 profile 0.3 与 `overlap_review`；0.5 为严格次要设置。配对单位始终是题目，不把 bbox 或模型调用当作独立样本。

主估计量：

`Delta Recall = Macro Necessary Recall(predicted-plan) - Macro Necessary Recall(repeated-single)`

使用 10,000 次 question-level paired bootstrap、固定 seed `20260903` 计算 percentile 95% CI。只有一个主比较，不做多重比较修正。

结论分三层：

1. **Coverage support**：predicted-plan 相对等图像调用次数 repeated-single 提高 Macro Necessary Recall。
2. **Efficient grounding support**：在 coverage support 基础上，Macro F1 同时提高且平均预测 union area 不增加。
3. **Planning bottleneck evidence**：保持相同 step 数时，reviewed-plan 优于 predicted-plan。

必须同时报告 Macro Precision、Macro F1、Complete Evidence Rate、Spatial Upper Bound、Mean Matched IoU、预测框数量、union area、图像调用次数、prompt/completion token 和时延。若 recall 上升同时 precision 下降且面积扩大，只能声称覆盖增加，不能声称 grounding 更高效或更可信。

## 8. 正式性与冻结门槛

`--formal` 会强制：

- 精确覆盖 annotation_1 的 41 个索引及冻结顺序；
- 协议 `FROZEN.json` 有效；
- 禁止 `--allow-unfrozen-plans` 和 engineering review；
- 记录 annotation、TSV、逐图像和所有输入来源 SHA-256。

正式运行只有在 `complete=true`、`protocol_complete=true`、41 题全覆盖且无 request failure 时才能执行 `--freeze-existing`。冻结还会再次核对协议版本、协议哈希、annotation 哈希和协议 manifest。之后用 `--verify-frozen` 检查文件缺失、内容变化和意外新增文件。

5 题 smoke 或 calibration 即使完成也不能冻结为正式结果。冻结资产只能通过新输出目录扩展，禁止覆盖 H1、single-pass 或历史 H2 结果。

## 9. 主要产物

每个运行目录包含：

- `experiment_config.json`
- `run_status.json`
- `plans.json`
- `step_outputs.json`
- `rig_results.json`（兼容现有 evaluator）
- `raw/qXXXX/plan.txt` 与逐 step 原始输出
- `samples/qXXXX/` 的 step bbox、union bbox 和元数据
- `visualizations/qXXXX_union.jpg`
- 正式冻结后的 `FROZEN.json`

协议目录包含精确 prompt/schema/指标口径的 `protocol.json`，以及同时哈希 protocol 与 runner 源码的 `FROZEN.json`。
