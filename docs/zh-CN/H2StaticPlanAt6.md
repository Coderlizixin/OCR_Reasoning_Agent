# H2 Adaptive Gate：Static Question-Only Plan@6

该实验是27B Human-Adaptive Gate的静态计划对照，不修改冻结的H2-v2。planner只看问题并必须输出恰好6个互不重复的可定位证据需求；随后grounder独立查看完整图像执行6步。所有视觉调用使用与 repeated-single@6 相同的图像处理参数、12288 completion-token预算、开启thinking、不显式设置temperature并使用guided JSON。

## 运行顺序

先生成5题静态计划：

~~~powershell
python test_H2_Static_Plan_at6_on_Charxiv.py --model-name Qwen3.5-27B --smoke --plan-only --output-dir test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__static-question-plan-at6__plan-only
~~~

若仅出现API/request failure，使用相同命令添加 --resume。truncated 或 parse_failed 是模型失败，不能补采样。只有5题全部产生有效6步计划时才允许冻结：

~~~powershell
python test_H2_Static_Plan_at6_on_Charxiv.py --plan-only --output-dir test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__static-question-plan-at6__plan-only --freeze-plan-existing

python test_H2_Static_Plan_at6_on_Charxiv.py --plan-only --output-dir test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__static-question-plan-at6__plan-only --verify-plan-frozen
~~~

再执行6步grounding：

~~~powershell
python test_H2_Static_Plan_at6_on_Charxiv.py --model-name Qwen3.5-27B --smoke --plans-from test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__static-question-plan-at6__plan-only/plans.json --output-dir test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__static-question-plan-at6
~~~

grounding中断后使用完全相同命令添加 --resume。

## 输出

最终并集保存在 rig_results.json；逐步原始输出保存在 step_outputs.json 与 raw/qXXXX/；prefix_results/k1 到 k6 分别包含Evaluator兼容的 rig_results.json，用于计算Recall@1–6。prefix_results.json 保存全部题目和全部前缀，visualizations/qXXXX_kN.jpg 保存每轮证据并集。

主比较必须使用同5题、相同最大6次视觉调用的 repeated-single@6。本条件仍是open-loop question decomposition，不是最终adaptive方法。
