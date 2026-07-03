# Acceptance Criteria

结构说明：Level 1 拆为 Must / Should / Optional；两个可投递检查点（Week 4 末、Week 6 末）；Week 5 为门控三选一。CUDA extension 不是并行任务线。

## Level 1: 主线

### Must（缺一不可，检查点 1 的底线）

Reference 与数据结构：

- PyTorch dense reference 可运行，作为 FP32 ground truth。
- PyTorch paged reference 可运行，与 dense 对齐。
- block-table generator：随机乱序、碎片化分配、未用 slot 填垃圾值。

Triton kernel：

- 支持 `q_len=1`。
- 支持 variable-length batch。
- 支持 block table。
- 支持 `head_dim=128`、FP16。
- kernel 核心循环第一稿手写（LLM 只 review / debug）。

Correctness tests：

- batch=1 / 多 batch。
- context length 小于一个 block / 刚好等于 block size / 跨多个 block。
- 最后一个 block 未满。
- block table 非连续、随机顺序。
- 未用 slot 填垃圾值时结果不变（验证不越界读）。
- FP16 tolerance，以 FP32 dense reference 为 ground truth。

Benchmark：

- 输出 CSV，图表由 CSV 自动生成。
- 主配置 sweep：context 128 → 16K 以上、batch 1 → 32。
- CUDA events 计时、warmup / repeat、p50 / p95。
- `nvidia-smi -lgc` 锁时钟或记录时钟状态；环境快照（GPU 型号 / driver / 版本）写入结果。
- baselines：PyTorch dense SDPA、PyTorch paged reference、FlashInfer decode wrapper。
- 有效带宽利用率：理论读取量 / 实测 latency，对比硬件峰值带宽的百分比。

Profiling report 初稿：

- memory-bound 论证（基于带宽利用率）。
- paged layout 间接寻址代价。
- online softmax 开销。
- 小 batch + 长 context 的 SM 占用问题。

### Should（时间允许则做，Week 6 前补齐）

- `head_dim=64`。
- BF16 及其 tolerance 测试。
- block size sensitivity sweep（8 / 16 / 32 / 64）。
- 32K 长 context 覆盖。
- NCU 深度分析（若 Week 0 验证权限可用）。
- 全套图表：latency vs context length、tokens/s vs batch size、block size sensitivity。

### Optional

- 完整 shape grid。
- roofline 图。

## 检查点 1（Week 4 末，第一个可投递状态）

- Level 1 Must 全部完成。
- README 能让外人看懂并复现主结果。
- `docs/benchmark-results.md`、`docs/profiling-report.md` 初稿存在。
- `RESUME_SNIPPETS.md` 检查点版本可直接放简历。

## CUDA 门控（Week 4 末判定）

开门条件（三条全部满足）：

1. Level 1 Must 全绿。
2. 检查点 1 文档完成。
3. lab notes 显示 kernel / profiling 阶段享受多于煎熬。

无论是否开门，交付 `docs/cuda-design-sketch.md`：

- 线程块到 (batch, head) 的映射。
- K/V block 的 shared memory staging。
- online softmax 的 warp shuffle 归约。
- 向量化加载（如 float4）。
- 与 Triton 自动处理部分的逐项对照。

## Level 2（Week 5，门控三选一，检查点 2）

### 默认: split-KV（Flash-Decoding 风格）

- context 分段并行计算 partial softmax，reduce kernel 合并。
- correctness 与 v1 kernel 全部测试对齐。
- batch=1/2 长 context 前后对比图。
- 报告解释 SM 占用率变化及收益边界。

### 门开且投算子岗: 最小 CUDA/C++ PyTorch Extension

- 范围锁死：`head_dim=128`、MHA、FP16。
- custom op 可 import，C++ binding 可编译，kernel 可 launch。
- 复用既有 correctness tests 与 reference 对齐。
- 与 Triton 版本做接口和性能对照。
- 不要求覆盖 benchmark grid，不要求超过 Triton。

### 兴趣转向 infra: Mini KV Block Allocator + Request Simulation

- 支持 request arrival / finish、block allocation / free。
- 输出 fragmentation、block reuse、memory usage trace。
- 能解释分页策略如何影响 serving 资源利用率。

## Level 3（Level 1 / 2 稳定后）

1. GQA / MQA（优先，成本低）。
2. FlashInfer / vLLM 设计层对照深化（benchmark baseline 已在 Level 1）。
3. INT8 KV cache（不作暑期主线）。

## 过程性验收（贯穿全程）

- 每周合卷复述：能独立手推 online softmax 的 running max / running sum 更新式与 block table 地址计算。
- 每周 lab note：最难的 bug、学到什么、享受 / 排斥什么（区分工具链痛苦与方向排斥）。
- README 与简历 snippet 每周更新，任何一周中断项目都是完整的小故事。
- 最终形态（Week 6 末）：README 第 16 节十个问题能独立回答；limitations 含 vAttention 反方证据。
