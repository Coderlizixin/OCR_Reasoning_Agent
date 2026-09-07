# H2 前置实验：GT Evidence-Need Oracle Grounding

## 目的

本实验不验证完整H2，而是验证其最低必要前提：当模型已经获得一个正确的证据需求时，当前grounder能否在图像中定位对应的必要证据。若该Oracle条件仍然失败，继续开发自动adaptive planner不会解决主要瓶颈。

## 输入与防泄漏

固定样本为`1,17,8,244,346`，共49条GT annotation。每次模型调用只输入原问题、完整图像和一条GT `description`，不输入GT bbox、答案或其他annotation。GT bbox仅在模型输出后用于空间评价。同一题内不存在重复description，因此目标身份没有由重复文本造成的歧义。

runner为`test_H2_Oracle_Grounding_on_Charxiv.py`，输出与冻结H1/H2目录隔离。thinking开启，temperature不显式修改，guided JSON开启，每次completion预算12288 token。`finish_reason=length`按模型失败计0，不补采样或增加token；只有API异常记为`request_failed`并允许`--resume`。

## 指标

评价单位是一条GT evidence-need请求。主命中规则与现有RIG profile一致：IoU达到阈值，或GT coverage至少0.7且预测框/GT面积比不超过4。另报告纯IoU命中、best IoU、预测框数量、大框率、模型失败和token/时延。

工程Go/No-Go门槛预设为：profile Recall@0.3至少60%，模型失败率不超过20%，request failure为0，且不能主要依赖大框命中。

## 9B五题结果

49条请求中44 valid、5 truncated、0 request failure。目标级Profile Recall为30.61%@0.3和24.49%@0.5，样本Macro Recall为31.48%@0.3和25.52%@0.5；mean best IoU为19.62%，模型失败率10.20%，大框-only率8.16%。

逐题Recall@0.3为：q1 30.00%、q17 33.33%、q8 75.00%、q244 19.05%、q346 0%。诊断分组中axis/tick为7/13，legend为0/11，line/region为5/12，point/cluster/peak为1/11。这里优先把含`line/cells`的描述归入line/region，避免`first coordinate`字样将曲线集合误归为point类。

同5题Direct-single参考的Macro/Micro Recall@0.3为25.83%/22.45%。Oracle分别只提升5.64/8.16个百分点，却使用49次而非5次调用，总token约48.56万而非5.45万。

## 结论

Recall门槛未通过，因此当前结论为`NO_GO_FOR_FULL_ADAPTIVE_SYSTEM_WITH_CURRENT_GROUNDER`。正确语义需求能带来小幅收益，但当前模型仍经常选错图例、曲线或面板，点/峰值定位偏移，并出现局部目标的大区域框选。主要瓶颈已经从“planner不知道找什么”定位到“grounder即使知道目标也难以精确定位”。

下一步应在同一49条固定请求上研究panel-aware候选生成、OCR/legend proposal、目标候选选择和bbox refinement。只有在不改变Oracle任务定义的情况下通过Recall门槛，才进入human adaptive planning和自动adaptive agent实验。
