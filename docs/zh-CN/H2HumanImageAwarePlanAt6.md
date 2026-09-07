# H2 Human-Image-Aware Plan@6

该条件用于检验一个严格受控的前置问题：当独立计划作者能够查看题目和原图、但不能查看GT、答案或任何模型输出时，六条实例感知的证据需求能否比等图像调用次数的`repeated-single@6`形成更好的证据覆盖与效率前沿。

它不是自动agent，也不包含re-embedding。它只隔离“图像感知的evidence-need planning”本身。

## 信息边界

计划作者只能查看authoring packet中的题目和未修改原图，不得查看：

- annotation及其description/bbox；
- 参考答案或模型答案；
- repeated、static、oracle的plan、bbox、可视化或裁决文件；
- 任何目标bbox或由标注转述而来的提示。

如果作者此前接触过上述信息，其计划必须另标为`GT-exposed/researcher-authored`，不能冒充本条件。当前主助手已经看过GT描述，因此不直接填写正式human计划；应由新的独立作者完成。

## 六步要求

每题恰好六条、有序且不重复的evidence need。每条应指向一个具体可见对象或同质对象集合，并能由grounder返回bbox。允许根据原图写出实际可见的subplot、series、label、marker、cluster或value身份，但不能给出bbox坐标，不能要求比较、排名、计数、计算、选择或最终答案。

步骤按预期边际证据价值排序。禁止仅用一步请求整张图或main chart panel；若题目涉及多面板，应直接点名与该步有关的具体面板及其中对象。完整六步应覆盖候选、身份映射、数值/轨迹和必要参照信息，而不是模板化填满结构角色。

## 创建盲审包

~~~powershell
python prepare_H2_Human_Image_Aware_Plan_at6.py create
~~~

输出目录为：

`test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate/Qwen3.5-27B__human-image-aware-plan-at6__authoring`

其中只包含5张原图、`authoring_context.json`、填写用`human_plans.json`和说明文件。将整个目录交给未接触GT和已有实验输出的独立作者。

## 校验与冻结

作者填写30条evidence need、匿名ID和全部信息访问声明后运行：

~~~powershell
python prepare_H2_Human_Image_Aware_Plan_at6.py validate
~~~

任何空步骤、重复步骤、问题/上下文哈希变化或不完整声明都会阻止通过。过宽panel/bbox式请求作为warning处理，必须修改后才能冻结。

~~~powershell
python prepare_H2_Human_Image_Aware_Plan_at6.py freeze
python prepare_H2_Human_Image_Aware_Plan_at6.py verify
~~~

冻结输出位于独立的`Qwen3.5-27B__human-image-aware-plan-at6__plan-only`目录，不修改Static、Repeated或GT Oracle资产。

## 比较口径

后续grounding继续使用Qwen3.5-27B、完整原图、thinking开启、temperature不显式设置、12288 completion-token和同一guided JSON。每题固定六次视觉grounding，与`repeated-single@6`严格匹配。

主报告使用完成语义裁决的Macro Necessary Recall@0.3，并同时报告Macro Precision/F1、Complete Evidence Rate、bbox数、union area、step新增命中、重复率、token和时延。由于Repeated可以通过大量框提高召回，还必须报告固定bbox预算和固定area预算下的Recall/Pareto结果。

## 执行冻结计划

独立执行入口为 `test_H2_Human_Image_Aware_Plan_at6_on_Charxiv.py`。
它只读取已冻结的Human计划，每步调用复用Static的grounding prompt、schema、
图像编码和processor配置；每一步不读取之前的输出，不调用planner。
模型固定为Qwen3.5-27B，completion预算固定为12288。

在项目目录执行一次q1检查：

~~~powershell
python test_H2_Human_Image_Aware_Plan_at6_on_Charxiv.py --indices 1
~~~

随后补齐五题：

~~~powershell
python test_H2_Human_Image_Aware_Plan_at6_on_Charxiv.py --smoke --resume
~~~

默认读取 `Qwen3.5-27B__human-image-aware-plan-at6__plan-only/plans.json`，
输出到同级 `Qwen3.5-27B__human-image-aware-plan-at6/`。可用
`--plans-from`、`--output-dir`、`--api-base`指定路径或服务。
首次运行可加 `--dry-run`，验证全部五题的冻结哈希与输入一致性，不调用API或写文件。

实验cohort始终为冻结的五题；`--indices`仅决定当前调用执行哪些题。
q1完成后 `invocation_complete=true`，但全五题的 `complete=false` 是正常状态。
`--smoke --resume`跳过已经保存的valid、truncated、parse_failed步骤，
仅执行缺失步骤并重试request_failed；不会丢弃其他题的结果或重新采样模型失败。
已有API故障及重试记录保存在 `prior_request_failures`。
冻结计划、模型、协议/源码、TSV、输入图像、endpoint变化时拒绝续跑。

输出包括配置、冻结计划副本、逐步记录、最终并集、k1–k6前缀并集、
原始文本、可视化、token/时延、source/request/encoded-image哈希。
每步完成即保存结果与run status。五题完成后应有30条step记录、
`complete=true`、`request_failed_samples=0`；模型失败单独计数，不因此删除题目。
此时只完成grounding执行，必要证据指标仍需独立语义裁决。

应在项目既有的 `ocrr-T` conda环境中运行。不要把项目内其他Python依赖目录
加入 `PYTHONPATH`，否则其中为不同Python版本编译的扩展可能覆盖conda环境依赖。

## 五题结果（2026-09-07）

27B五题运行完成：30/30 step valid，0 truncated、0 parse failure、0 request
failure。Human语义裁决包含44个重叠候选，42 true、2 false、0 pending；两项
false均为q1红/蓝曲线之间的错误系列空间重叠。候选key哈希已固化。

在`overlap_review`、阈值0.3下，Human@6的Macro Recall/Precision/F1为
75.98%/41.94%/53.40%，平均19.6框、union area 21.54%，Complete Evidence
Rate为0。Repeated@6对应80.02%/14.12%/23.93%、58.4框、29.11%面积；
Static@6为75.93%/28.00%/38.66%、34.8框、53.13%面积。

Human与Static的Recall差为+0.05 pp（五题paired bootstrap 95% CI
[-15.00,+17.05] pp），与Repeated差为-4.05 pp（[-15.00,+2.86] pp）。
因此五题点估计支持Human相对Static形成更好的Recall–Precision–Area组合；
相对Repeated则是以少量Recall换取显著更少的框、更小面积和更高Precision/F1。
样本过少，不能据此作总体显著性结论。

Human前缀k1–k6的Macro Recall依次为29.90%、42.24%、49.10%、62.29%、
73.12%、75.98%；Macro F1为45.14%、52.56%、52.12%、55.62%、61.66%、
53.40%。第六步仅增加2.86 pp Recall，却使平均框数14.6→19.6、Precision
55.83%→41.94%，说明当前固定六步的最后一步边际效率较低。

主要miss为：q1/q8遗漏GT定义的完整x-axis；q17底部点簇框过窄且派生坐标点
未被直接定位；q244底排三个完整subplot框相对GT曲线带过宽，未通过面积比限制；
q346第一峰被框成整幅subplot，x-axis label与ticks合框后不满足匹配。完整reviewed
结果为运行目录的`evaluation_report_reviewed.json`，三组汇总与哈希清单位于
adaptive-gate根目录的`Qwen3.5-27B__adaptive-gate-5-human-reviewed-summary.json`
和`Qwen3.5-27B__adaptive-gate-5-human-reviewed-manifest.json`。
