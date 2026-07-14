# Learning Syllabus

这份文档是项目的教学大纲。它不替代 `ROADMAP.md`：`ROADMAP.md` 负责项目交付节奏，
这里负责每一课怎么学、学完要会讲什么、要写什么、用什么验证。

大纲按“阶段”组织，避免把学习体验绑定到固定周数。实际推进时可以根据状态快慢调整，
但每个阶段结束前都要留下代码、测试、笔记和一次口头复述。

## 0. 使用方式

每一课都按同一个节奏推进：

1. 先明确本课目标和输入输出。
2. 再解释核心概念、shape（形状）、layout（布局）和 correctness（正确性）标准。
3. 用户先写核心学习代码的第一版。
4. Agent 做 review（审查）、debug（调试）和项目风格整理。
5. 补 semantic tests（语义测试）、boundary tests（边界测试）和学习笔记。
6. 最后做一次问答复述，确认这个阶段可以讲清楚。

非核心学习内容，例如测试脚手架、笔记整理、benchmark harness（基准测试框架）和
plotting（画图），可以由 Agent 更主动补齐。

## 1. Reference 阶段：把正确性钉牢

### 目标

用最直接的 PyTorch FP32 实现定义 decode attention（解码注意力）的正确答案，并证明
paged KV layout（分页 KV 布局）和 dense KV layout（稠密 KV 布局）在语义上等价。

### 要掌握的概念

- attention（注意力）里 `Q/K/V` 的角色。
- prefill（预填充）和 decode（解码）的区别。
- decode 阶段为什么通常是 `q_len = 1`。
- KV cache（KV 缓存）为什么像“前缀和”：用空间换时间，避免重复计算历史上下文的 K/V。
- `B/H/S/D` 的含义：
  - `B`: batch（批次）
  - `H`: head（注意力头）
  - `S`: sequence/context length（序列/上下文长度）
  - `D`: head_dim（单个 head 的维度）
- `context_lens` 如何定义每个 batch item 的有效上下文长度。
- `block_tables` 如何把 logical block（逻辑块）映射到 physical block（物理块）。

### 代码任务

- 实现 `dense_decode_attention`。
- 实现 `paged_decode_attention`。
- 实现或整理 `make_random_block_tables`。
- 补齐 dense reference、paged-vs-dense 和 block table generator 的测试。

### 正确性标准

- dense reference 使用 FP32，作为后续所有实现的 ground truth（真值）。
- paged reference 输出需要和 dense reference 对齐。
- 未使用 slot（槽位）填入 garbage values（垃圾值），确认不会读到无效位置。
- 覆盖 batch=1、多 batch、单 block、跨 block、最后一个 block 未填满、随机 physical block 顺序。

### 阶段产出

- `src/paged_kv_attention/reference.py`
- `src/paged_kv_attention/block_table.py`
- `tests/test_reference.py`
- `tests/test_reference_contracts.py`
- `note/reference-stage-attention-kv-cache.md`
- `note/dense-decode-attention-implementation.md`
- `note/reference-stage-testing.md`

### 复述问题

- dense reference 为什么可以作为真值？
- `context_lens` 为什么不能只做 shape check？
- paged reference 为什么要和 dense reference 对齐，而不是直接和 Triton kernel 对齐？
- `logical_block = t // block_size` 和 `slot = t % block_size` 分别表示什么？
- garbage slots 能抓到哪类 bug？

## 2. Triton 入门阶段：先会写最小 kernel

### 目标

理解 Triton programming model（Triton 编程模型），先能写、运行、测试简单 kernel，
再进入 attention kernel（注意力内核）。

### 要掌握的概念

- program instance（程序实例）和 `program_id`。
- block-level parallelism（块级并行）。
- `tl.arange`、mask load/store、`tl.load`、`tl.store`。
- stride（步长）和 pointer arithmetic（指针算术）。
- 为什么 GPU kernel 里要显式处理边界 mask。

### 代码任务

- 跑通一个 vector add kernel（向量加法内核）。
- 写一个 row-wise operation（按行操作）的最小 kernel，例如 row sum 或 row max。
- 给最小 kernel 补 correctness tests（正确性测试），和 PyTorch 结果对齐。

### 正确性标准

- CPU/PyTorch reference 与 Triton 输出一致。
- 覆盖长度不能整除 block size 的情况。
- mask 不应读写越界元素。

### 复述问题

- `program_id(0)` 通常代表什么？
- `tl.arange(0, BLOCK)` 生成的是数据，还是 offset（偏移）？
- mask load 为什么经常需要 `other=`？
- Triton kernel 的并行单位和 PyTorch tensor operation 有什么不同？

## 3. Softmax 阶段：从普通 softmax 到 online softmax

### 目标

先写清楚普通 softmax，再推导 online softmax（在线 softmax）。这是进入 FlashAttention
和 decode attention kernel 的核心数学桥梁。

### 要掌握的概念

- 数值稳定 softmax：

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

- block-wise max / sum（分块最大值/求和）。
- running max（运行中最大值）和 running sum（运行中归一化分母）。
- 为什么新的 block 可能改变全局最大值，并导致旧分母需要 rescale（重缩放）。

### 推导目标

处理一个新 block 时：

```text
m_new = max(m_old, max(scores_block))
l_new = l_old * exp(m_old - m_new) + sum(exp(scores_block - m_new))
```

如果同时维护 weighted value accumulator（加权 V 累加器）：

```text
acc_new =
    acc_old * exp(m_old - m_new)
    + sum(exp(scores_block - m_new) * v_block)
```

最终输出：

```text
out = acc_final / l_final
```

### 代码任务

- 用 PyTorch 写一个普通 softmax attention。
- 用 PyTorch 写一个 block-wise online softmax attention。
- 让 online 版本和 dense reference 对齐。

### 正确性标准

- 覆盖多个 block。
- 覆盖 scores 差距很大的数值稳定场景。
- 输出与 dense FP32 reference 在合理 tolerance（容差）内一致。

### 复述问题

- 为什么不能每个 block 各自 softmax 后直接相加？
- `m_old - m_new` 为什么一定小于等于 0？
- running sum 为什么要 rescale？
- online softmax 和普通 softmax 结果为什么等价？

## 4. FlashAttention 桥接阶段：理解它和本项目的关系

### 目标

理解 FlashAttention 和本项目的共同点与差异：两者都关心 attention 的 memory traffic
（内存访问量）和 softmax 的分块计算，但本项目重点是 decode 阶段的 paged KV cache。

### 要掌握的概念

- FlashAttention 主要优化 prefill / training-like attention 中的大 `Q x K^T` 矩阵。
- 本项目当前目标是 decode attention，`q_len = 1`，主要瓶颈是读取长上下文 KV cache。
- 两者共同依赖 online softmax。
- FlashAttention 的分块思想可以帮助理解 Triton kernel 的循环结构，但不能直接照搬到
  paged decode attention。

### 阅读任务

- 只读 FlashAttention / FlashAttention-2 里和 block-wise softmax、online softmax 相关的部分。
- 读 Triton fused attention 教程时，重点看 block loop 和 accumulator，不照搬 prefill 结构。

### 复述问题

- FlashAttention 主要减少的是哪类中间矩阵的显存读写？
- decode attention 为什么没有大的 `Q x K^T` 矩阵要 materialize（物化）？
- 本项目为什么仍然需要 online softmax？
- FlashAttention 的哪些思想可以迁移，哪些不能直接迁移？

## 5. Triton Decode v0：连续 KV layout，不分页

### 目标

先在连续 dense KV layout 上写一个 Triton decode attention kernel，把 online softmax 写对。
这一阶段暂时不加入 block table，降低调试维度。

### 接口范围

- `q`: `[B, H, D]`
- `k/v`: `[B, S, H, D]`
- `context_lens`: `[B]`
- `out`: `[B, H, D]`
- 初始限制可以先锁定：
  - `q_len = 1`
  - MHA（multi-head attention，多头注意力）
  - `head_dim = 128`
  - FP16 input，FP32 accumulation（累加）

### 代码任务

- 一个 program instance 负责一个 `(batch, head)`。
- 顺序扫 context blocks。
- 在 kernel 内维护 running max、running sum 和 accumulator。
- 输出和 `dense_decode_attention` 对齐。

### 正确性标准

- batch=1 和 multi-batch。
- `context_lens` 不同。
- context length 小于、等于、大于一个 block。
- 输出与 dense reference 对齐。

### 复述问题

- 一个 Triton program 对应哪个输出元素区域？
- 每轮循环读入的 `k_block/v_block` shape 是什么？
- accumulator 的 shape 是什么？
- 为什么 v0 先不加 block table？

## 6. Triton Paged v1：加入 block table

### 目标

把连续 KV 读取替换成 paged KV 读取。算法数学不变，主要变化是地址计算。

### 要掌握的概念

- logical block 到 physical block 的映射。
- `slot = t % block_size` 在 physical block 内定位 token。
- non-contiguous（非连续）和 random-order（随机顺序）physical blocks。
- paged layout 对 fragmentation（碎片）的改善，以及对间接寻址的额外开销。

### 代码任务

- 在 Triton kernel 中根据 `block_tables[b, logical_block]` 找 physical block。
- 从 `paged_k/v[physical_block, slot, h, d]` 读取 K/V。
- 复用 v0 的 online softmax 主循环。
- 与 `paged_decode_attention` 和 dense reference 对齐。

### 正确性标准

- block table 顺序连续、非连续、随机顺序都通过。
- garbage slots 不影响输出。
- variable-length batch（变长 batch）通过。
- FP16/BF16 输出相对 FP32 dense reference 在 tolerance 内。

### 复述问题

- 加入 block table 后，数学公式变了吗？
- 哪些 bug 是 dense layout 测不出来的？
- paged KV 降低了什么浪费，又引入了什么成本？

## 7. Benchmark 与 Profiling 阶段：讲清楚性能故事

### 目标

证明 Triton paged kernel 不只是正确，还能在合理场景下快过 PyTorch paged reference，并能解释性能瓶颈。

### 要掌握的概念

- benchmark（基准测试）和 correctness test（正确性测试）的区别。
- warmup（预热）、repeat（重复次数）、p50、p95。
- CUDA events（CUDA 事件）计时。
- analytical bytes read（解析法读取字节数）和 effective bandwidth（有效带宽）。
- memory-bound（内存受限）判断。
- profiling（性能剖析）工具：NCU、nsys、`torch.profiler`。

### 代码任务

- benchmark 输出 CSV。
- 覆盖不同 batch、head、context length、block size。
- 加 baseline（基线）：PyTorch dense SDPA、PyTorch paged reference、FlashInfer、Triton implementation。
- 生成图表和性能说明。

### 正确性标准

- benchmark 前先跑 correctness tests。
- 记录环境事实：GPU、driver、CUDA、PyTorch、Triton、关键包版本。
- 只比较相同输入、相同 dtype、相同输出语义的实现。

### 复述问题

- 为什么 benchmark 不能代替 correctness test？
- p50 和 p95 分别说明什么？
- effective bandwidth 怎么估算？
- 小 batch 长 context 为什么可能 SM occupancy（SM 占用率）不足？

## 8. Triton Split-KV 阶段：用 context 并行解决小 batch 利用不足

### 目标

把一个 `(batch, head)` 顺序扫描完整 context 的工作拆成多个 `(batch, head, split)`
programs，计算 partial online-softmax state，再通过 reduce kernel 合并最终输出。

### 要掌握的概念

- program saturation curve（程序数量饱和曲线）。
- 为什么 program 数少于 SM 数不等于必然低效，为什么带宽饱和点必须实测。
- partial `m/l/acc` 的语义和可结合合并规则。
- split 数增加带来的并行收益与 intermediate state / reduce 成本。
- 为什么 split-KV 只适合小 batch、长 context，而不是无条件启用。

### 接口范围

```text
partial_m:    [B, H, num_splits]
partial_l:    [B, H, num_splits]
partial_acc:  [B, H, num_splits, D]
out:          [B, H, D]
```

### 代码任务

- 实现 Triton split partial kernel。
- 实现 Triton split reduce kernel。
- 支持 `split=1/4/8/16`。
- 实现基于 batch、heads、context 和 saturation evidence 的 adaptive dispatch。
- 生成 single-pass / split-KV 前后对比图。

### 正确性标准

- 所有 split 配置与 FP32 dense reference 对齐。
- 覆盖 variable-length batch、random-order block table、partial final block 和 garbage slots。
- `split=1` 与 single-pass paged kernel 语义一致。
- reduce 合并顺序不改变合理 tolerance 内的结果。

### 复述问题

- partial `m/l/acc` 为什么足以表示一段 context？
- 两个 partial state 如何合并？
- 为什么 `split=32` 不一定比 `split=16` 快？
- adaptive dispatch 需要观察哪些 workload 参数？

## 9. CUDA/C++ Port 阶段：手动实现已验证的 single-pass kernel

### 目标

将已经通过 correctness 和 benchmark 验证的 single-pass paged decode attention 移植为
限定范围的 CUDA/C++ PyTorch extension，理解 Triton 自动完成了哪些代码生成与调度工作。

### 范围边界

```text
q_len:       1
attention:   MHA
head_dim:    128
input dtype: FP16
accumulate:  FP32
layout:      paged KV + block table
```

CUDA split-KV 不属于主线验收，只作为 stretch goal。

### 要掌握的概念

- C++ binding 与 PyTorch extension build。
- CUDA grid/block/thread mapping。
- shared-memory staging（共享内存暂存）。
- warp shuffle / block reduction。
- vectorized loads 与对齐要求。
- register pressure、occupancy 与手动 launch configuration。

### 代码任务

- 先完成 `docs/cuda-design-sketch.md`。
- 实现 extension binding 和 kernel launch。
- 实现 block-table 寻址、tail mask 和 online softmax。
- 复用现有 reference tests。
- 与 Triton single-pass kernel 做接口、延迟和有效带宽对照。

### 正确性标准

- extension 可构建、import 和 launch。
- 覆盖 batch=1/multi-batch、variable context、random-order block table 和 partial tail。
- FP16 output semantics 与 FP32 reference 在 tolerance 内对齐。
- 不要求 CUDA 超过 Triton，但必须解释性能差异。

### 复述问题

- Triton program instance 在 CUDA 中对应什么执行组织？
- 哪些 reduction 在 Triton 中是一行、在 CUDA 中需要显式协作？
- shared memory、register 和 global memory 分别保存什么？
- 为什么先移植 single-pass，而不同时移植 split-KV？

## 10. 打包阶段：把项目变成可讲述作品

### 目标

把代码、测试、benchmark、profiling 和学习笔记整理成一个面试可讲的完整项目。

### 产出

- 更新 README 的项目故事和结果。
- 更新 `RESUME_SNIPPETS.md`。
- 写 `docs/benchmark-results.md`。
- 写 `docs/profiling-report.md`。
- 写 `docs/cuda-design-sketch.md`。
- 补 limitations（限制）和 future work（后续工作）。

### 复述问题

- 这个项目优化的瓶颈是什么？
- 为什么选择 paged KV？
- 为什么先写 reference？
- online softmax 怎么保证数值稳定？
- Triton kernel 的 program mapping 是什么？
- 性能结果说明了什么，没说明什么？
- split-KV 的收益边界是什么？
- CUDA 与 Triton 各自承担了哪些手动/自动优化工作？

## 11. 当前进度记录

当前已完成：

- dense/paged FP32 reference 与 block-table generator；
- block-wise online softmax 推导与 PyTorch reference；
- Triton dense/paged single-pass decode kernels；
- random-order block table、garbage slots 与 variable-length correctness tests；
- CUDA-event benchmark harness、CSV、图表与 program saturation experiment。

当前正在收束 performance checkpoint：

1. hardware peak bandwidth utilization；
2. PyTorch paged reference baseline 与 FlashInfer quantitative baseline（定量基线）；
3. `torch.profiler` 与 `docs/profiling-report.md`；
4. benchmark 笔记、lab note 与 Git checkpoint。

随后进入 Triton split-KV，不并行开始 CUDA。
