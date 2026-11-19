# A1 第一段集中复习：端到端模型与训练架构

这份练习记录 2026 年 8 月的复习范围。11 月已从下一 token 预测重新开始回顾，
最新学习状态和下一步见 [A1 进度](../../../progress/a1.yaml) 的 `current`。

2026-08-29 复习范围调整：学习者自报已掌握任务 1、任务 2 的前三项和任务 3，
不再要求重复书写；这些目前记为“自报掌握，未独立复核”。当前补讲缺口为任务 2 的
byte fallback/跨分词器 loss 边界，以及任务 4–7。补讲结束后只会使用三道综合判断
进行缺口确认。

## 这一段在课程中的位置

这份复习题整合的是 A2–A5 都会继续使用的基础链路：

```text
原始文本 → 分词器 → token 批次 → Transformer → 损失 → 梯度/参数更新
         → 检查点/评估/生成 → 关于行为和质量的证据
```

对应原始 handout 位置：

- BPE 动机：`docs/handouts/cs336_assignment1_basics.md:108-220`（PDF 第 3–5 页，§2）
- Transformer 总览：`docs/handouts/cs336_assignment1_basics.md:552-626`（PDF 第 13–14 页，§3.1）
- RMSNorm、FFN、RoPE、attention：
  `docs/handouts/cs336_assignment1_basics.md:857-1205`（PDF 第 19–26 页，§3.4）
- 交叉熵、AdamW、学习率调度、梯度裁剪：
  `docs/handouts/cs336_assignment1_basics.md:1280-1562`（PDF 第 28–34 页，§4）
- 数据、检查点与训练循环：
  `docs/handouts/cs336_assignment1_basics.md:1563-1677`（PDF 第 34–37 页，§5）

可以查阅四份 `notes/a1/*-knowhow.md` 笔记。本次目标是把知识连起来，不是闭卷背诵。

---

## 任务 1：追踪一个批次经过完整系统

使用以下符号：`B=批次大小`、`T=上下文长度`、`D=模型宽度`、`H=注意力头数`、
`V=词表大小`。

从一维的序列化 token 流开始，写出每个编号位置的张量形状，并说明该位置引入了什么新信息
或监督信号：

```text
token 流
  → (1) 采样得到输入 x 和错开一位的目标 y
  → (2) token embedding
  → (3) 拆分成多头后的 Q、K、V
  → (4) attention 分数
  → (5) 一个 Transformer block 对外输出
  → (6) LM head 输出
  → (7) 交叉熵输出
```

然后解释：训练为什么能并行监督全部 `T` 个位置，同时又不会把每个位置的目标 token 泄漏给
对应的预测。

### 你的回答

<!-- 在这里填写任务 1 的完整答案。 -->


---

## 任务 2：分词器同时改变统计问题和系统成本

假设两个 byte-level BPE 分词器在**同一份原始语料**上评估：

| 分词器 | 词表大小 | 平均每个 token 包含的原始字节数 |
| --- | ---: | ---: |
| A | 10,000 | 4.0 |
| B | 32,000 | 4.5 |

不要预设 B 一定更好。请解释：

1. 对同样的文本，哪个分词器产生的 token 更少？大约少多少比例？
2. 这会如何改变 attention mixing 的成本？其中与序列相关的项正比于 `T²D`。
3. 这会如何改变 embedding/LM head 的参数量，以及词表投影的计算量？
4. 为什么 byte fallback 能避免 OOV（词表外 token），却不能保证每种字符串或领域都能得到
   同样好的表示？
5. 为什么使用不同分词器或不同数据集训练的模型，不能直接拿原始 validation loss 做质量比值？

写出近似比例并把权衡解释清楚即可。

### 你的回答

<!-- 在这里填写任务 2 的完整答案。 -->


---

## 任务 3：定位通信、局部计算与位置信息

对下面这个 pre-norm block：

```text
z = x + Attention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

请用一段连贯解释覆盖以下内容：

- 哪个操作负责在 token 位置之间通信，哪个操作只在每个位置内部变换特征？
- residual path 与 RMSNorm 分别为优化提供什么？
- 为什么 block 对外保持 `(B,T,D)`，而 attention 内部会构造 `(B,H,T,T)`？
- 为什么带 causal mask 的 NoPE 模型经过多层后并非完全看不懂顺序？
- 对只包含相同 token 表示的前缀，边界情况究竟是什么：为什么 softmax attention 无法区分
  3 个相同 value 与 30 个相同 value？RoPE 又如何打破这个对称性？

### 你的回答

<!-- 在这里填写任务 3 的完整答案。 -->


---

## 任务 4：区分三种不同的尺度控制机制

假设进入子层的 residual 向量被乘以一个正标量 `c`。

1. 为什么 `RMSNorm(cx)` 近似不变？请同时考虑 epsilon 和可学习 gain。
2. 为什么移除 RMSNorm 后，Q/K 分数、FFN activation、logits 和梯度会对 residual 尺度漂移
   更敏感？
3. 一个无 normalization 的训练虽然每一步原始梯度范数都是 `10⁵`，但都被裁剪到 1，loss
   因而仍是有限值。这算稳定优化吗？梯度裁剪恢复了什么，又没有恢复什么？
4. 降低 peak LR 后，同一个无 normalization 模型成功训练。它补偿了什么？仍然缺少哪条
   normalization 不变量？
5. 为什么延长 warmup 不能保证一个不变且本来就不安全的 peak LR 最终变得安全？

### 你的回答

<!-- 在这里填写任务 4 的完整答案。 -->


---

## 任务 5：重建一次优化器更新和一次可信续训

把下面事件排成正确的因果顺序，并解释其中两个关键边界为什么重要：

```text
AdamW 更新       backward       清空梯度       计算 forward logits
全局梯度裁剪      计算交叉熵      选择当前 schedule LR      采样错位批次
```

- 为什么 clipping 必须发生在 backward 之后、AdamW 更新 moment estimates 之前？
- 如果使用四个 gradient-accumulation microbatch，loss 除法/梯度累积、裁剪、AdamW 更新和
  scheduler 前进一步分别在什么时候发生？请区分 physical batch 与 effective batch。
- 一个中断的训练只重新加载模型权重，然后在相同名义 iteration 继续。为什么它可以推理，
  却不是 trajectory-equivalent 的续训？还必须恢复哪些状态？
- `best.pt` 与 `last.pt` 分别回答什么问题？

### 你的回答

<!-- 在这里填写任务 5 的完整答案。 -->


---

## 任务 6：根据证据诊断，但不要过度下结论

对下面每种现象，给出第一判断以及下一项能够区分原因的检查。请把四项当作一份完整的
调试报告，而不是互不相关的小题。

1. 一个新模型训练数百次更新后，仍无法把同一个固定 minibatch 的 loss 降到 4 以下。
2. training loss 持续下降，但使用固定样本评估的 validation loss 开始上升。
3. training/validation loss 都是有限值，但裁剪前梯度范数反复从 `0.5` 跳到 `10⁴`，几乎每次
   更新都被裁剪，并且 validation loss 来回震荡。
4. validation loss 平滑改善，但采样文本只有局部语法，仍有虚构事实、实体不一致和重复循环。

对于第 4 项，请区分哪些问题可能来自模型学到的分布，temperature/top-p 又能改变什么。
不要把 decoding 当成凭空补充模型知识的手段。

### 你的回答

<!-- 在这里填写任务 6 的完整答案。 -->


---

## 任务 7：用真实 A1 证据审计结论

对每条结论标记为：**证据支持**、**证据排除**或**尚未建立**，然后用下方证据说明理由。
即使实验成功，如果结论范围过大，也不能标记为“证据支持”。

现有证据：

- 固定 minibatch loss：`4.102376 → 0.000080`。
- 连续训练 8 步，对比 `4 步 + checkpoint + 4 步`：最大参数差为 `0.0`。
- TinyStories validation：`9.2500 → 1.6903`；生成故事语法基本成立，但有时重复或语义矛盾。
- OWT validation：`10.3862 → 4.6485`；样本学到了网页/新闻形式，但会虚构实体和主张。
- 无 normalization、peak LR `3e-3` 时严重发散；同一结构使用 `1e-3` 时达到 validation `3.556`。

需要审计的结论：

A. 实现好的 forward/backward/update 链能够拟合给定的 next-token 目标。\
B. 当前 checkpoint 格式对任何设备、精度和未来配置都保证 trajectory-equivalent。\
C. TinyStories loss `1.6903` 证明它的语言模型质量优于 OWT loss `4.6485`。\
D. validation loss 下降证明模型变得事实正确并且全局连贯。\
E. 在这次浅层模型实验中，RMSNorm 扩大了可用学习率范围。\
F. 数学上，任何 Transformer 在任何学习率下都必须有 RMSNorm 才能训练。

最后，用三句话向一名工程师交接：目前已知系统能够做到什么、仍有什么行为限制、在提出
更强结论之前还需要什么证据。

### 你的回答

<!-- 在这里填写任务 7 的完整答案。 -->


---

## 2026-08-29 约定的确认方式

当时约定把书面复习缩减为三道综合判断题，只覆盖学习者指出的缺口，补讲后再集中审阅。
