# TTS-CIR 下一步执行蓝图（基于 arXiv:2602.23029）

日期：2026-04-23  
目标：将当前想法快速推进到“可投递主线”状态（优先 CVPR/ICCV）

## 1) 一句话研究主张（可直接用于摘要）

我们将训练自由 CIR 重构为**隐变量边缘化**问题：通过测试时并行采样编辑假设并进行结构化约束重排，在相同预算下优于串行 refinement，并呈现可复现的 test-time scaling 增益。

---

## 2) 先做什么：两周 MVP（先证伪再扩展）

> 原则：先验证“是否存在 scaling law”，再堆机制。

### Week 1：Scaling Existence Check（最低成本）

- 数据：CIRR val（主）、FashionIQ val（辅）
- 模型：固定 CLIP ViT-L/14（冻结）
- 变量：`K ∈ {1,2,4,8,16,32}`
- 采样：
  - Textual-hypothesis（温度重写 + 模板改写）
  - Latent-direction（embedding 小扰动）
- 汇总：先用最简单 `max_k sim_k`（不加结构化评分）

**必须产出的图：**
1. `Oracle Recall@100 vs K`
2. `R@1/R@5 vs K`
3. `WallClock/query vs K`

**Gate A（继续/停止阈值）：**
- 若 `Oracle Recall@100` 在 `K=8` 相比 `K=1` 提升 < 2 个点：暂停；重做 `p(z|T)` 设计。
- 若 `R@1` 随 K 完全不升：说明采样噪声过高或汇总失效。

### Week 2：机制最小化验证（结构化评分）

在固定 `K=8` 下只引入两个机制：
1. 结构化多维评分（object/attribute/style/violation）
2. 负约束惩罚（`S = S_pos - λS_neg`）

**Gate B：**
- `Ours-full` 必须显著优于 `K=8 + 单相似度`，否则 C2 不成立。

---

## 3) 实验与审稿防线（最关键）

### 3.1 公平预算协议（必须双口径）

所有主表都同时报告：
- `ForwardPass/query`
- `WallClock/query`

并给出：
- Equal-FWD 对齐结果
- Equal-Time 对齐结果

> 审稿人最常见攻击：你只是“算得更多”。双口径是必要防线。

### 3.2 基线收敛策略（最多三家族）

- Family A：WISER（主基线）
- Family B：推理增强训练自由法（可复现优先）
- Family C：检索+重排训练自由法

若某方法复现失败：
- 记录 commit、配置与失败原因；
- 用同家族可复现实例替换，并在附录披露。

### 3.3 统计显著性

- seeds=3
- query-level paired bootstrap（95% CI）
- 主指标：R@1/R@5/R@10

---

## 4) 结构化评分的“可落地实现”

## 4.1 评分分解

对候选图像 `I` 与假设 `z_k`：

- `s_obj`: 对象一致性
- `s_attr`: 属性匹配
- `s_style`: 风格兼容
- `s_neg`: 负约束违背

总分：

`S(I) = max_k [ w_obj s_obj + w_attr s_attr + w_style s_style - λ s_neg ]`

### 4.2 建议实现（训练自由）

- `s_obj/s_attr`: 文本短语（noun/adj phrase）与候选图像的 CLIP 匹配
- `s_style`: style prompt bank（如 minimal / elegant / vintage）匹配
- `s_neg`: 对“not/without/remove”短语构造反向提示并取匹配惩罚

### 4.3 参数选择

- 开始用网格：`w_obj:w_attr:w_style = 1:1:0.5`, `λ ∈ {0.2,0.5,1.0}`
- 在 CIRR val 固定后冻结，避免测试集调参嫌疑。

---

## 5) 你当前方案中最容易被挑刺的点（及应对）

1. **“只是候选池更大”**  
   - 反证：固定 K，比 `simple max sim` 与 `structured score`。
2. **“依赖强 LLM 才有效”**  
   - 反证：template/lightweight 采样达到 strong LLM 的 ≥95%。
3. **“串行法也能靠堆叠追上”**  
   - 反证：给 Pareto 前沿图（性能-时延/前向）。

---

## 6) 论文写作顺序（推荐）

1. 先写 Figure 3（Scaling 曲线）与 Table 1（公平预算主结果）
2. 再写 Table 2（机制隔离）
3. 最后补失败分桶与可视化（增强可信度）

> 先把“能否公平领先 + 能否随 K 扩展”讲透，投稿胜率会高很多。

---

## 7) 一页式执行清单（可直接跟踪）

- [ ] 跑通评测与预算日志一致性（M0）
- [ ] 锁定可复现最强三家族基线（M1）
- [ ] 产出 `K` 扩展主曲线（M2）
- [ ] 完成机制隔离 + 简洁性 Pareto（M3）
- [ ] 完成失败分桶与附录扩展（M4）

