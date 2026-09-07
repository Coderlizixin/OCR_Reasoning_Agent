---
name: ocr-reasoning-current-research
description: 当前 OCR agent 与 Step-QA 研究线、实现差异和已知陷阱
metadata:
  type: project
  last_verified: 2026-09-04
  commit: "6282752"
---

# 当前主线

最近提交的重心是两条相互关联的研究线：

1. OCR-Agent 的 Planning → Executor → Reasoning 分阶段设计，重点分析 grounding/understanding 对最终推理的影响。
2. 从 OCR-Reasoning 推理过程构造 Step-QA 和依赖式 Step-QA，用于区分单步能力、依赖传播和最终综合错误。

最近一次提交还包含 CharXiv grounding 分析，因此 `test_RIG_on_Charxiv.py`、`_test_descriptive_q1.py`、`test_case/RIG_Charxiv/` 及 RIG 标注工具属于当前诊断上下文。

# 当前 RIG 必要线索基线

- 正式人工标注只使用 `RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json`：41 个样本、303 个必要线索；`annotation_0.json` 是工具测试用例，必须排除。
- 当前 annotation v1 已足够完成第一阶段验证：每个 annotation `id` 作为一个独立必要语义线索，不要求先升级 schema v2。
- 15/41 个样本存在 GT bbox 相交，共 53 对。评估不能逐 GT 独立取最大 IoU，否则一个预测框可能重复命中多个必要线索。
- `evaluate_rig_grounding.py` 与 `rig_eval/` 实现一对一空间匹配、重叠样本语义裁决、Necessary Recall、Complete Evidence Rate、Union Region Coverage、预测框数量/面积和答对/答错条件汇总。
- 默认 `overlap_review` 策略：非重叠样本自动采用一对一空间匹配；重叠样本必须完成 `semantic_review.json` 裁决，pending 样本不混入正式语义指标。论文最终结果可改用 `review_all` 全量人工语义裁决。
- `evaluate` 可通过 `--answer-scores <xlsx>` 按 `index` 连接外部答案评分表的 `score` 列；bbox解析失败样本仍保留为证据召回0，同时根据外部score进入答对/答错条件汇总。
- RIG多模型实验统一隔离在 `test_case/RIG_Charxiv/runs/<API模型名>/`。历史27B过程文件快照位于 `runs/Qwen3.5-27B/`；新的9B实验写入 `runs/Qwen3.5-9B/`。推理脚本默认拒绝写入非空运行目录，且逐样本增量保存结果与状态，避免覆盖或中断丢失。
- 27B的答案条件指标必须连接 `outputs/Qwen3_5-27B-API/Qwen3_5-27B-API_CharXiv_reasoning_val_qwen-plus.xlsx`，不能使用9B评分。规范报告是根目录的 `test_case/RIG_Charxiv/evaluation_report.json`，快照副本是 `runs/Qwen3.5-27B/evaluation_report_corrected_27B_scores.json`；快照内原始 `evaluation_report.json` 仅作为错误score来源的历史记录保留。
- Direct BBox当前实验版本为 `derivation-evidence-v4-concise-guided-json`：优先使用不含答案格式指令的 `raw_question`；为避免9B在删除检验、最小证据集和反事实讨论上耗尽token，prompt直接要求定位下游推理所需证据。仍禁止只框答案标签；比较题必须覆盖尺度/判据、每个候选观测和标签/图例映射。vLLM `guided_json`继续约束JSON，thinking和temperature保持服务器现状。默认目录为 `runs/<模型名>__derivation-evidence-v4-concise-guided-json/`，resume会拒绝混入不同prompt版本结果。
- 完整操作和指标口径见 `docs/zh-CN/RIGEvidenceEvaluation.md`。

# RIG 后续研究计划（2026-08-28）

## Pilot结论与任务定义

- 最新语义裁决已完成：9B为27个true、5个false，27B为70个true、8个false；两组均为41 finalized / 0 pending。项目内8月24日生成的旧报告早于最新裁决，正式引用前必须重新生成并另存final-reviewed版本。
- 在IoU/复合空间阈值0.3下，9B/27B的Macro Necessary Recall分别为26.65%/56.07%，Macro Precision为32.86%/48.48%，Complete Evidence Rate为2.44%/14.63%；答题准确率却达到78.05%/85.37%。这初步确认“答案正确不等于完整获取必要视觉证据”。
- 正式任务暂定为Question-Conditioned Necessary Visual Evidence Grounding：输入text-rich图像和问题，在不依赖最终答案的情况下，输出由`description+bbox`组成的原子视觉证据集合；集合应覆盖参考推导路径所需事实、定位准确且不过度冗余。当前v1评估的是Reference Necessary Evidence Set，不宣称人工标注是唯一可能的逻辑证据集合。
- 27B相对9B在Recall、Precision、Complete Rate和定位质量上均显著提升，说明任务具有初步可解性；但27B在宽松阈值下也仅6/41题证据完整，任务远未解决。
- Actual Recall与Spatial Upper Bound的差距低于1个百分点，当前主要瓶颈是必要证据漏检、定位偏差或多个线索合并，而不是空间候选框的语义裁决错误。

## 分阶段计划清单

### P0：固化可重复基线

- 用最新裁决重新生成9B/27B正式报告，保存为不会覆盖历史文件的final-reviewed版本。
- 导出逐题配对表，包含GT/预测数量、Recall、Precision、IoU、重叠属性、题型和外部answer score。
- 固化prompt、模型/API名、max tokens、thinking、temperature、图像处理参数以及annotation/预测/裁决文件哈希。
- 对Macro Recall、Precision和Complete Rate计算sample-level paired bootstrap 95%置信区间，并用配对bootstrap或置换检验验证27B相对9B的提升。

### P1：验证任务构念有效性（最高优先级）

- GT evidence-only：下游模型只看问题和GT裁剪，验证其准确率能否接近完整原图。
- Predicted evidence-only：只看模型预测证据，衡量证据的功能充分性，并与GT上限比较。
- Evidence deletion：在原图遮挡GT证据，与完整原图、同面积随机遮挡和非必要区域遮挡比较答案下降。
- Answer-only control：只提供最终答案文字区域，证明答案定位不能替代推导证据。
- 至少增加第二标注者，完整复标15个重叠样本并抽查10–15个非重叠样本；报告证据匹配F1、bbox一致性和语义裁决一致率。

### P2：验证泛化性与稳健性

- 将数据扩展到100–200题后再稳定规范，目标benchmark规模500题以上；按图像来源划分train/dev/test并冻结test。
- 平衡四类inst_category、GT证据数量、重叠比例、图像类型和推理难度。
- 增加至少两个其他VLM家族，并保留9B/27B规模对照。
- 进行prompt、guided JSON、thinking、分辨率和重复运行消融；所有模型统一max tokens、温度、bbox格式和重试策略。
- annotation v1继续服务pilot；先用约50题试验v2，再决定是否加入evidence role、step/group、alternative evidence set、required/optional和依赖关系。

### P3：建立方法与基线体系

- 无能力基线：随机框、中心框、整图框、同面积随机区域、答案区域、全部OCR文字框。
- Direct VLM基线：当前v4、answer-then-ground、evidence-first、checklist-then-ground及thinking开关。
- 新方法优先采用有限结构化流水线：Evidence Requirement Planner → OCR/图表元素候选生成 → Question-conditioned Selector → Coverage Verifier → bbox refiner。
- Planner只输出有限证据角色，不展开自由CoT；Verifier只返回缺失类型，避免9B无限反思和token耗尽。
- 重点提升Spatial Upper Bound，同时以Precision、预测框数量和面积比例防止大框/多框退化。

### P4：功能评价、错误分析与论文验证

- 主指标：Macro/Micro Necessary Recall、Macro Precision、Complete Evidence Rate；辅助指标：Spatial Upper Bound、Mean Matched IoU、Union Coverage、预测数量/面积。
- 功能指标：GT/Pred Evidence-only Accuracy、Sufficiency Gap、Evidence Deletion Drop及area-matched random control。
- 建立错误标签：无bbox、只框答案、遗漏候选/尺度/图例映射、合并线索、框过大/截断、身份/OCR错误、证据完整但推理错误、token耗尽、JSON失败。
- 论文核心论证链：答案正确不代表证据完整 → 定义独立任务 → 验证标注充分性/必要性 → 跨模型展示缺口 → 提出结构化证据发现方法 → 证明其改善grounding与下游推理。

## 近期执行顺序

1. 生成并固化最新final-reviewed报告及逐题配对表。
2. 完成GT evidence-only、GT deletion和随机区域对照，证明任务本身有效。
3. 完成第二标注者一致性实验。
4. 扩展数据和模型家族，确认现象普遍性。
5. 实现Evidence Planner + proposal + coverage verifier，超越单轮prompt基线。

## GT Mask-In 前期可行性实验（2026-08-31）

- 已新增 `test_GT_MaskIn_on_Charxiv.py`，对 annotation v1 的 GT bbox 构造像素级并集 mask：保留 GT 区域、其余区域填充固定灰色，同时保持原图画布和空间坐标不变。bbox 重叠或完全覆盖不会重复暴露像素，但各语义 annotation 仍分别保存在元数据中。
- 模型实际输入不绘制 bbox、不包含 annotation description；默认沿用 CharXiv 的 `question` 字段，以便与标准 full-image 答案准确率比较。thinking 不关闭，temperature 默认与当前 API 配置一致为 0.0，均写入实验配置。
- 默认41题按模型隔离写入 `test_case/RIG_Charxiv/gt_mask_in/<model>__gt-mask-in-v1/`，支持 `--prepare-only`、`--dry-run`、`--resume`、502重试和显式 `--api-base`。
- 过程文件包括模型实际接收的PNG、二值mask、逐题JSON、run status、配置与annotation哈希，以及兼容 `CharXiv.evaluate` 的Excel；不完整运行默认禁止judge，避免把基础设施失败误计为模型错误。
- GT Mask-In只能验证GT证据的功能充分性/去干扰潜力，不能单独证明learned re-embedding有效。正式前期论证仍需与同配置Full Image、同面积Random Mask-In、GT Mask-Out和learned/predicted evidence进行配对对照。
- 运行命令、输出解释和公平性要求见 `docs/zh-CN/GTMaskInExperiment.md`。
- 2026-09-01完成9B/27B各41题推理。HTTP层均为41/41成功，但固定12288 completion-token预算下，9B有8题、27B有5题以`finish_reason=length`结束；应视为无有效答案，而不是普通成功。脚本现已区分HTTP完成与Valid Answer，并支持`--rebuild-existing`从原始JSON无损生成`predictions_normalized.json`、`predictions_for_judge.xlsx`和`answer_quality_report.json`；原始过程文件不覆盖。
- 同日修复CharXiv judge的部分缓存续评错误：原实现使用pending keys但仍提交全量tasks，导致`track_progress_rich`断言失败；现按pending indices同时构造keys/tasks。`OpenAIWrapper`也不再在日志中打印完整API Key。
- 27B首次judge又暴露评测入口误选旧`predictions.xlsx`的问题：截断题空单元格被Pandas读为float NaN，在`str.replace`处触发TypeError。现已改为优先使用`predictions_for_judge.xlsx`，并在`auxeval`中将NaN安全归一为空字符串，防止单题异常中止整个并发评测。
- 规范化judge结果为：9B GT Mask-In 29/41（70.73%），其中8题超过固定12288 token预算；27B GT Mask-In 34/41（82.93%），其中5题超预算。超预算是模型在固定计算预算下的能力失败，主指标直接计0，不再通过提高max tokens补跑；Valid Answer Accuracy只作为失败模式诊断。
- 2026-09-01已固化两组正式GT Mask-In目录：各目录新增带逐文件SHA-256的`FROZEN.json`，9B记录95个文件、27B记录93个文件且首次校验均通过。`test_GT_MaskIn_on_Charxiv.py`会拒绝对冻结目录执行推理、resume、重建和重新评测，新实验不得覆盖这些目录。
- 同一runner现支持`full-image`、`gt-mask-out`、`random-mask-in`和`gt-roi`。Random Mask-In通过GT联合mask的确定性循环平移精确保留像素面积与形状，并在固定候选中最小化GT重叠；GT ROI输入顺序为“GT Mask-In布局图 → annotation顺序的逐框ROI → question”，不提供annotation description。
- 已用indices 1、17、8、244、346覆盖四类题目和4–21个GT框的范围，对五种condition完成`prepare-only`检查，目录位于`test_case/RIG_Charxiv/evidence_conditions/smoke-5/`。所有25组图像的像素变换、随机面积、ROI数量、路径、可解码性和多图消息顺序均通过`--validate-prepared`，报告保存在各condition的`preparation_validation_report.json`。
- 2026-09-02已完成并固化9B condition矩阵：Full Image、GT Mask-Out、三个Random Mask-In种子和GT ROI分别记录90、90、131、131、131、393个文件；加上既有GT Mask-In 95个文件，共7个来源、1061个文件，逐文件SHA-256均复验通过。合并哈希清单位于`test_case/RIG_Charxiv/evidence_conditions/condition_hash_manifest.json`。
- 9B统一结果为：Full Image 29/41（70.73%）、GT Mask-In 29/41（70.73%）、GT Mask-Out 11/41（26.83%）、Random三种子分别5/41、1/41、1/41（均值5.69%）、GT ROI 25/41（60.98%）。主比较中`Full−GT Mask-Out=+43.90 pp`（paired bootstrap 95% CI [24.39, 63.41]，exact McNemar p=2.77e-4），`GT Mask-In−Random mean=+65.04 pp`（95% CI [51.22, 78.86]），而`GT Mask-In−Full=0 pp`（95% CI [-17.07, 17.07]）。
- 统一报告由`analyze_evidence_conditions.py`生成JSON、`build_condition_comparison_workbook.mjs`生成公式驱动的XLSX；正式文件为`condition_comparison_report.json/xlsx`，包含逐题配对、置信区间、McNemar检验、题型统计和来源哈希。生成27B报告时必须显式指定不同输出文件名，禁止覆盖9B报告。
- 2026-09-02完成27B condition矩阵的工程闭环：Full Image、GT Mask-Out、三个Random Mask-In和GT ROI分别冻结91、91、132、132、132、394个文件；连同既有GT Mask-In 93个文件，共7个来源、1065个文件。所有目录均为41/41题、`complete=true`、0 request failure，逐目录`--verify-frozen`均通过。六个新版目录的输入文件、像素变换与多模态顺序全量校验通过；历史GT Mask-In缺少`model_input_manifest.json`，已只读验证41题图像/像素变换，但历史请求顺序不能独立复核。
- 27B统一结果为：Full Image 34/41（82.93%）、GT Mask-In 34/41（82.93%）、GT Mask-Out 15/41（36.59%）、Random三种子分别8/41、1/41、5/41（均值11.38%）、GT ROI 31/41（75.61%）。主比较中`Full−GT Mask-Out=+46.34 pp`（95% CI [31.71, 60.98]，exact McNemar p=3.81e-6），`GT Mask-In−Random mean=+71.54 pp`（95% CI [57.72, 84.55]），`GT Mask-In−Full=0 pp`（95% CI [-12.20, 12.20]）。27B独立报告和哈希清单位于`evidence_conditions/reports/Qwen3.5-27B/`。
- 9B/27B跨模型报告位于`evidence_conditions/reports/cross_model_9B_27B/`，以同一41题做配对bootstrap和McNemar检验，不把82个观测当作独立样本。跨模型Full、GT Mask-In、GT ROI分别提升12.20、12.20、14.63 pp；H1的关键condition effect在两个规模上方向一致。原有9B JSON/XLSX/哈希清单保持原路径和原哈希，不被跨模型报告覆盖。
- 冻结入口现要求`run_status.json`中的`complete=true`、请求索引全覆盖且无request failure；新版`FROZEN.json`原因包含实际模型与condition。对已经冻结的目录执行`--validate-prepared`只做只读校验，不写入冻结资产。

## H2：Planning 是否改善必要证据发现（2026-09-03）

- H2的目标不是再次证明GT evidence有效，而是在相同问题与图像上检验分解是否使证据并集覆盖更多必要线索：比较single、额外调用次数对照repeated-single、自动predicted-plan和人工reviewed-plan。H2只评grounding，不在本阶段加入最终回答、GT step监督或re-embedding。
- 独立runner为`test_H2_Multistep_Grounding_on_Charxiv.py`，输出根目录为`test_case/RIG_Charxiv/h2_grounding/`，会拒绝写入既有`test_case/RIG_Charxiv/runs/`。single模式只读导入`derivation-evidence-v4-concise-guided-json`；其他模式写入独立目录并逐题增量保存。
- 四种模式统一生成`experiment_config.json`、`plans.json`、`step_outputs.json`、evaluator兼容的`rig_results.json`和`run_status.json`；API模式还保存每个plan/step原始文本、step bbox、union bbox及provenance、token、时延、finish reason和可视化。旧single产物没有时延时必须显式标为unavailable，不能伪装成0秒。
- thinking保持开启，temperature不显式修改，plan/step/single每次调用统一使用12288 completion-token预算。早期3072/4096 smoke会让thinking耗尽预算后没有闭合JSON，相关诊断目录保留但不得用于H2比较。
- 9B固定smoke indices为`1,17,8,244,346`。最终single、2x repeated-single、predicted-plan、engineering-only reviewed-plan均为5/5样本完成、0 request failure；现有`rig_eval.core.load_predictions`可直接读出全部5个index。自动计划逐题产生5/5/4/6/4步；reviewed engineering计划为5/4/4/6/4步。
- smoke的IoU 0.5纯空间诊断显示single/repeated/predicted/reviewed的Macro Recall约为19.33%/29.69%/43.90%/53.33%，但Macro Precision约为24.00%/21.34%/14.77%/14.05%，平均预测union面积比例约为8.18%/16.08%/35.56%/33.91%。这只说明管线可运行并暴露明显over-grounding，不构成H2结论。
- `repeated-single`正式比较必须通过`--repeat-counts-from <predicted plans.json>`逐题匹配自动计划的图像调用次数；当前2x smoke只做格式检查。reviewed smoke文件明确为engineering-only，正式实验必须由研究者在不查看答案、GT bbox和annotation description的条件下重新审核。
- H2单元测试位于`tests/test_h2_multistep_grounding.py`，覆盖计划解析、模型端点映射、matched repeat counts、token归一化、重叠框并集面积、语义重复保留、review approval和现有evaluator loader兼容性。smoke汇总与限制见`test_case/RIG_Charxiv/h2_grounding/smoke-5/smoke_validation_report.json`。

### H2 正式协议冻结（2026-09-04）

- 正式版本为`h2-grounding-v2-concise-complete`。精确prompt、guided JSON schema、41题顺序、调用预算、失败计分、review约束、并集规则、统计指标和结论门槛写入`docs/zh-CN/H2GroundingProtocol.md`及冻结目录`test_case/RIG_Charxiv/h2_grounding/protocols/h2-grounding-v2-concise-complete/`。
- `FROZEN.json`已生成且`--verify-protocol`通过；正式dry-run确认annotation、TSV与41张输入图像的SHA-256均已记录，协议SHA-256为`04e3fbfbcdaf63b679792ce9b43da5c26f9eb363096bcfa6fd8c7f0ca9a3d577`。冻结后禁止修改runner，否则校验会失败并阻止正式运行。
- 正式计划必须先以`--formal --plan-only`一次性生成并冻结。计划解析失败或截断按零步骤保留，不能重生成或人工修补；其matched repeated-single和reviewed-plan同题也必须执行零次图像调用。reviewed-plan只能在保持step id、顺序和数量不变的前提下改写evidence need，禁止接触图像、答案、GT或grounding结果。
- single模式会把既有H1预测按哈希复制为H2自己的只读输入快照；repeated-single必须逐题匹配predicted-plan的有效计划步数。正式冻结要求41题完整、0 request failure、`protocol_complete=true`，且协议、annotation和输入资产哈希全部一致。
- 最终9B五题工程校准使用indices `1,17,8,244,346`，计划步数为`4/4/3/6/3`，共20个step调用：15 valid、4 truncated、1 parse_failed，5/5样本均无request failure。纯空间诊断在IoU 0.3下Macro Recall/Precision/F1为12.24%/6.78%/8.16%，平均19.6个框、union面积12.06%；这是未做语义裁决的小样本工程诊断，不是H2科学结论，也不能据此继续调prompt。
- 旧v1 smoke虽有更高召回，但包含compare/conclude等答案式步骤，违反evidence-only构念，因此不作为正式协议。最终prompt按科学有效性与结构覆盖锁定，而不是按五题分数选择。完整校准决策保存在`test_case/RIG_Charxiv/h2_grounding/calibration-v2/protocol_calibration_report.json`。
- 下一步依次执行：9B正式41题predicted plan-only并冻结 → 基于同一plans运行predicted-plan与matched repeated-single → 盲审生成reviewed plan并运行 → 导入single快照 → 完成全量语义裁决、配对bootstrap和题型/失败分析；之后再在27B复现同一协议。

### H2构念修正与Oracle前置实验（2026-09-04）

- 冻结的v2属于question-only open-loop decomposition：planner不看图，无法知道实例中的候选数、图表结构或多面板布局，因此不能作为“视觉adaptive planning有效”的主验证，只保留为static question-only消融。41题v2正式运行暂缓，不能覆盖或删除冻结资产。
- 在开发完整adaptive agent之前先设置GT Evidence-Need Oracle Gate：模型每次只获得question、完整图像和一条GT annotation description，不获得GT bbox；逐GT测试当前grounder在已知正确需求时能否定位目标。独立runner为`test_H2_Oracle_Grounding_on_Charxiv.py`。
- 9B固定5题共49条Oracle请求已完成：44 valid、5 truncated、0 request failure；Profile Recall@0.3目标级30.61%、样本Macro 31.48%，@0.5分别24.49%/25.52%，mean best IoU 19.62%。逐题@0.3为q1 30.00%、q17 33.33%、q8 75.00%、q244 19.05%、q346 0%。
- 修正类别归类优先级后，诊断显示legend 0/11、line/region 5/12、point/cluster/peak 1/11、axis/tick 7/13。可视化复核确认主要是选错图例/曲线/面板、峰值或点框偏移及局部目标大框化，不是坐标缩放错误。
- 同5题Direct-single Macro/Micro Recall@0.3为25.83%/22.45%；Oracle只提升5.64/8.16 pp，却用了49次调用和约48.56万token（Direct为5次、约5.45万token）。预设Oracle Recall@0.3≥60%的门槛未通过，当前决策为`NO_GO_FOR_FULL_ADAPTIVE_SYSTEM_WITH_CURRENT_GROUNDER`。
- 下一步不是实现完整adaptive planner，而是在相同49条Oracle请求上改进target-conditioned localization：优先panel-aware候选生成、OCR/legend proposals、候选选择和bbox refinement。只有在Oracle协议不变时通过门槛，才开展human adaptive planning、real/no/shuffled feedback因果对照及后续re-injection实验。完整报告见`docs/zh-CN/H2OracleGroundingFeasibility.md`与Oracle运行目录的`feasibility_conclusion.json`。

### Adaptive Gate静态@6实现（2026-09-04）

- 27B Oracle前置实验达到目标级Profile Recall@0.3 71.43%、样本Macro 78.26%，47 valid、2 truncated、0 request failure，通过预设60% Gate；9B未通过，因此固定6次视觉调用的Adaptive Gate先只在27B开展。
- Adaptive Gate先比较repeated-single@6与static-question-plan@6，之后再实现human-adaptive@6。独立静态runner为test_H2_Static_Plan_at6_on_Charxiv.py：planner只看问题且必须输出恰好6步，grounder图像预处理与repeated-single一致，并导出k=1到6的Evaluator兼容前缀并集。
- 静态计划生成与grounding分离。5题计划全部valid后必须生成并验证PLAN_FROZEN.json，grounding运行只能读取该冻结plans.json。运行说明见docs/zh-CN/H2StaticPlanAt6.md。
- 2026-09-05完成27B五题`repeated-single@6`与`static-question-plan@6`的`overlap_review`逐项语义裁决。仅q1/q8/q244需要语义判断：Repeated为131个候选、113 true、18 false；Static为63个候选、57 true、6 false；两组均5 finalized、0 pending。错误系列重叠、标签/点替换和用单条曲线替代GT要求的完整曲线集合记false；能覆盖对应完整曲线集合的受控subplot区域记true，但其宽泛性继续由union-area惩罚。
- 裁决后Repeated/Static的Macro Recall@0.3分别为80.02%/75.93%，Static差值-4.10 pp，配对question bootstrap 95% CI为[-18.00,+5.71] pp；Macro Precision为14.12%/28.00%，Macro F1为23.93%/38.66%，平均bbox数58.4/34.8，但union面积29.11%/53.13%，Complete Evidence Rate均为0。Static未通过coverage support，也未通过要求面积不增加的efficient-grounding support；它只表明语义分工减少了框数量并提高空间精确率，尚未证明question-only planning有效。
- 可复核裁决脚本为`adjudicate_H2_adaptive_gate_5.py`，会校验候选key哈希并拒绝在候选集变化时写入；统一结果与8文件哈希清单位于`h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__adaptive-gate-5-reviewed-summary.json`和`Qwen3.5-27B__adaptive-gate-5-reviewed-manifest.json`。这仍是5题calibration，不得标为41题formal-run freeze。
- 下一步不扩展question-only static到41题，而是在相同5题、相同6次视觉grounding预算上构建`human-image-aware-plan@6`或`GT-grouped-evidence-plan@6`，验证实例感知的具体证据需求能否形成更好的Recall/Precision/Area Pareto前沿；若人工图像计划也不能改善，则暂缓完整adaptive agent并优先改进grounder。
- `human-image-aware-plan@6`的盲审协议与authoring packet已建立：`prepare_H2_Human_Image_Aware_Plan_at6.py create`只导出5题问题和原图，`validate/freeze/verify`强制六步、信息访问声明、上下文哈希和冻结manifest；说明见`docs/zh-CN/H2HumanImageAwarePlanAt6.md`。由于当前主助手已接触GT描述，正式30条evidence need必须由未看GT、答案、既有plan/bbox/可视化的独立作者填写；当前`human_plans.json`空模板未冻结是预期状态，不能用助手生成内容冒充盲审human条件。
- 2026-09-07收到独立作者首版30条human plan。结构完整、上下文哈希一致、无重复，但作者ID与5项访问声明尚未填写；语义复核还发现q8使用“highest/second highest”、q244使用“greatest increase”、q346预写“first/second peak”，均把派生结果编码进grounding need，q17还请求未显式标记的派生坐标位置。因此当前不能冻结。校验器已增加这些result-oriented phrase错误和derived-coordinate warning，7项human-plan测试通过；应由原作者基于不含GT的反馈自行修改后重新validate。
- 作者270已完成修改与信息访问声明；最终`human_plans.json`校验为valid、0 error、0 warning，上下文SHA-256保持`25d23372...656a`，计划源SHA-256为`34b25a7e...a5f`。独立plan-only目录已冻结，`prepare_H2_Human_Image_Aware_Plan_at6.py verify`检查3个文件、0 mismatch。该条件是human image-aware open-loop plan@6：作者看过原图后一次性给出六步，但执行时不根据前一步grounding反馈调整，因此不能称为closed-loop adaptive agent。

## Human@6 runner与q1检查（2026-09-07）

- 已实现独立 `test_H2_Human_Image_Aware_Plan_at6_on_Charxiv.py`，只读取冻结Human计划，固定27B、六次open-loop grounding及12288 completion-token，复用Static的prompt/schema/图像处理。未修改既有Static/Repeated/Oracle runner或结果。
- 实验cohort固定五题，`--indices 1`仅执行q1，随后 `--smoke --resume`保留q1并补齐其他题。valid/truncated/parse_failed不重采样，request_failed可续跑且保留此前故障记录。每步增量保存状态、原始输出、请求/图像哈希、并集与前缀结果。
- q1真实API运行已完成：6 valid、0 truncated、0 parse_failed、0 request_failed；各步4/4/4/8/3/1框，最终24框，union area 22.8553%，39197 total tokens，累计请求383.85秒。六个请求哈希与冻结需求/编码图像一致，k1–k6并集及evaluator兼容性通过，冻结Human计划复验通过。
- 工程检查报告为 `h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__human-image-aware-plan-at6/q1_validation_report.json`。可视化存在重复图例定位，尚未做语义裁决，不将本次工程成功作为H2效果结论。
- 三组相关测试共15项通过（新runner 3项含完整mock续跑集成测试、Static 5项、Human计划7项）。五题 `--smoke --resume --dry-run`通过，用户将手动执行剩余indices 17/8/244/346共24次调用。当前全cohort `complete=false`、q1 `invocation_complete=true`是预期状态。
- runner应在既有 `ocrr-T` conda环境运行。一次Codex工程测试曾创建为Python 3.13编译的 `.local-runtime`；它与`ocrr-T`的Python 3.11不兼容，现已移除。不要将其他Python依赖目录注入该环境的`PYTHONPATH`。完整命令见 `docs/zh-CN/H2HumanImageAwarePlanAt6.md`。
- 2026-09-07 Human@6五题运行完成：30/30 valid，0模型/API失败。44个重叠候选完成语义裁决（42 true、2 false、0 pending），两项false为q1红/蓝曲线错误系列重叠，候选哈希由`adjudicate_H2_human_gate_5.py`锁定。
- Reviewed @0.3结果：Human Macro Recall/Precision/F1为75.98%/41.94%/53.40%，19.6框、21.54% union area；Repeated为80.02%/14.12%/23.93%，58.4框、29.11%；Static为75.93%/28.00%/38.66%，34.8框、53.13%。Human−Static Recall为+0.05 pp（paired bootstrap 95% CI [-15.00,+17.05]），Human−Repeated为-4.05 pp（[-15.00,+2.86]）；Complete Evidence Rate三者均0。
- Human前缀Recall k1–k6为29.90/42.24/49.10/62.29/73.12/75.98%，F1为45.14/52.56/52.12/55.62/61.66/53.40%。第六步仅增2.86 pp Recall，却使平均框14.6→19.6、Precision 55.83%→41.94%，最后一步边际效率较低。
- 五题点估计支持Human相对Static形成更好的Recall–Precision–Area组合；相对Repeated是少量Recall换取显著更少框/面积和更高Precision/F1，尚不能称严格Pareto支配。主要miss来自完整axis与label粒度、q17底部cluster框过窄及派生坐标构念、q244底排subplot大框、q346峰值定位退化为整subplot。结果仍是calibration，下一步应先做未见题的小规模确认并明确直接可见证据/派生坐标及curve-set框粒度规范，再决定闭环adaptive实现。

# OCR-Agent 版本关系

## V4

- 四类步骤：`ground-understand`、`transcribe`、`compute`、`logical`。
- Planning 输出 strategy 与满足“依赖只能指向更早步骤”的 DAG。
- Executor 按顺序/依赖执行；每阶段最多 3 次尝试。
- `ground-understand` 可走 locate → crop → understand，也可由 `skip_locate=True` 直接看整图抽取。
- 顶层 visual step 可做交叉验证；V4 仍保留失败 visual step 的 executor 层 re-plan。
- Reasoning 区分视觉观察（FACT）和计算/逻辑派生（DRAFT），输出 JSON 后格式化为推理文本与 `<a>answer</a>`。

## V5

- 研究意图是把失败后的 executor re-plan 改为 planning 层拆分，并放松计划粒度。
- 注册的 9B/27B V5 默认 `skip_locate=True`、`cross_validate=True`，所以常用 V5 实际绕过 bbox locate；要研究显式定位，使用 `Qwen3.5-9B-OCRAgentV5-Grounding-Understanding`（`skip_locate=False`）。
- 源码/配置注释称“planning failure 后递归拆分（最大深度 2）”，也保留了 decomposition prompt，但当前 `generate_inner()` 在 planning 失败时直接停止，未调用递归拆分。这是文档与实际控制流的已知漂移，修改时以控制流为准。
- Stage 3 是 text-only：不向最终 reasoning 再传原图。

## V6

- 基本执行器与 V5 接近，主要变化在 Stage 3。
- 最终 reasoning 会收到原图，但提示要求只做 gross-error sanity check；视觉观察仍是主要事实来源。
- 增加“单次推理、立即提交、避免自我怀疑/反复回看”的约束，目的是减少长输出导致 JSON/答案丢失。
- 已注册为 `Qwen3.5-9B-OCRAgentV6`，默认同样 `skip_locate=True`；应视为 V5 的实验变体，而不是已经验证替代所有 V5 实验的稳定版本。

# 共同实现约定

- 所有阶段通过 `_Base64Agent` / `OpenAIWrapper` 调 OpenAI-compatible endpoint。
- 结构化输出依赖 JSON Schema/guided JSON；解析器还会尝试从 markdown fence 或混杂文本中恢复 JSON。
- bbox 使用 `[x1, y1, x2, y2]` 的千分比归一化坐标，裁剪前换算到原图像素。
- `MAX_ATTEMPTS = 3` 是 stage/step 内部尝试总数，与 wrapper 注册中的 `retry=10`（HTTP/API wrapper 重试）不是同一层重试。
- trace 写到各注册项的 `trace_dir/execution_traces.jsonl`，并含 planning、executor、reasoning、耗时和 overall status；诊断优先读 trace，再读散落日志。
- 主要失败状态包括 planning parse/API/invalid、grounding locate/understand、transcribe、compute、logical、reasoning 和 unexpected execution error。

# Step-QA 数据链

典型链路如下：

1. `step_qa_pipeline.py` 从 OCR-Reasoning 结果 Excel 解析中英文步骤，分类并生成可验证 QA；LLM 失败时可模板回退。
2. `step_qa_generate.py` 可对中间 Excel 分段补生成 QA，`verify_step_qa.py` 用于检查产物。
3. `build_step_reasoning_qa_tsv.py` 生成 `outputs/step_qa/step_reasoning_qa.tsv` 和图片引用。
4. `Step_Reasoning_QA` 在一张原图上按顺序提问多个子问题，以“全部步骤正确”作为题目级成功标准，并另算单步指标。
5. `OCR_Reasoning_StepQA` 把前 `n-1` 个 step QA 作为提示注入原题，刻意排除最后一步以防最终答案泄漏。
6. `build_depended_step_qa.py` / `build_depended_tsv.py` 构造带上游依赖提示的版本；`Depended_Step_Reasoning_QA` 和 `evaluate_depended_serial.py` 用于分析依赖传播。

这些脚本的默认输入输出路径绑定具体实验轮次，运行前必须检查文件常量或 CLI 参数，不能假设默认路径仍存在。

# 已知陷阱与优先排查顺序

- `README.md` 当前为空，这是提交历史中的显式删除结果；项目说明应先看本记忆、`CLAUDE.md` 和代码，而不是自动“恢复” README。
- `CLAUDE.md` 中把 `pgr_ocragent.py` 称为 V4 legacy，与当前 `ocr_agent_v4.py`–`v6.py` 演化线不完全一致；以注册项和实现为准。
- V5/V6 常用注册默认跳过 locate。看到 grounding 效果异常时，先确认实际模型名和 `skip_locate`，再分析 bbox。
- planning 失败不会进入 Stage 3；空答案首先查 planning trace，而不是只查 reasoning prompt。
- Step-QA 的绝对 Windows 路径只是本机候选，移植失败时先查 `outputs/step_qa/` 与 `LMU_DATA`。
- 原始 OCR-Reasoning judge 当前强制串行；“推理很快但评测卡住”可能只是 judge 单进程，不一定是死锁。
- 当前工作树常有大量实验日志删除、图片变化和未跟踪数据。任何清理/提交前都要限定文件范围。
