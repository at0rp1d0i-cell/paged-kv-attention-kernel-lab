# Acceptance Criteria

结构说明：项目按四个串行 checkpoint 推进：performance baseline、Triton split-KV、CUDA runtime 与 final delivery。CUDA extension 只在 Triton split-KV 稳定后开始，不与核心 Triton 开发并行。

## Stage 1: Performance Baseline

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
- baselines：PyTorch dense SDPA、PyTorch paged reference、FlashInfer decode wrapper；若 FlashInfer 与固定 GPU/CUDA stack 不兼容，必须保留可复现 probe、版本组合和原始错误，且不得声称已完成定量对比。
- 有效带宽利用率：理论读取量 / 实测 latency，对比硬件峰值带宽的百分比。

Profiling report 初稿：

- memory-bound 论证（基于带宽利用率）。
- paged layout 间接寻址代价。
- online softmax 开销。
- 小 batch + 长 context 的 SM 占用问题。

### Should（时间允许则在 Final Delivery 前补齐）

- `head_dim=64`。
- BF16 及其 tolerance 测试。
- block size sensitivity sweep（8 / 16 / 32 / 64）。
- 32K 长 context 覆盖。
- NCU 深度分析（若 Week 0 验证权限可用）。
- 全套图表：latency vs context length、tokens/s vs batch size、block size sensitivity。

### Optional

- 完整 shape grid。
- roofline 图。

## Performance Checkpoint（第一个可复现状态）

- Stage 1 Must 全部完成。
- README 能让外人看懂并复现主结果。
- `docs/benchmark-results.md`、`docs/profiling-report.md` 初稿存在。
- README 明确记录实现范围、关键结果、复现方法和已知限制。

## Triton Split-KV Checkpoint

- context 分段并行计算 partial softmax，reduce kernel 合并。
- correctness 与 v1 kernel 全部测试对齐。
- 覆盖 `split=1/4/8/16`、variable-length batch、random-order block table 与 partial tail。
- batch=1/2 长 context 前后对比图。
- adaptive dispatch 基于 batch、head、context 与 saturation evidence 选择 single/split path。
- batch=16/32 不因错误启用 split 而产生明显回退。
- 报告解释 program 数、有效带宽、reduce 开销与收益边界。

## CUDA Runtime Checkpoint

开始条件：

1. Performance checkpoint 完成。
2. Triton split-KV correctness 与性能报告完成。
3. `docs/cuda-design-sketch.md` 明确线程映射、归约、staging 与 vectorized loads。
4. CUDA extension smoke toolchain 仍可构建运行。

最小 CUDA/C++ PyTorch Extension：

- 范围锁死：`head_dim=128`、MHA、FP16。
- 实现 single-pass paged decode attention，不默认移植 split-KV。
- 支持 variable-length batch、block table 与 tail mask。
- custom op 可 import，C++ binding 可编译，kernel 可 launch。
- 复用既有 correctness tests 与 reference 对齐。
- 与 Triton 版本做接口和性能对照。
- 不要求覆盖 benchmark grid，不要求超过 Triton。
- CUDA split-KV 仅作为 stretch goal。

## Final Delivery

- GQA / MQA 作为 Should 项，时间允许则补齐。
- FlashInfer / vLLM 设计层对照深化。
- 最终 README、benchmark/profiling 报告与 CUDA design sketch 的接口、结果和限制一致。
- limitations 包含 vAttention 反方证据、解析带宽模型限制和未实现的 serving allocator。
- INT8 KV cache 不作为当前项目主线。

## 过程性验收（贯穿全程）

- 每周合卷复述：能独立手推 online softmax 的 running max / running sum 更新式与 block table 地址计算。
- 每周 lab note：最难的 bug、学到什么、享受 / 排斥什么（区分工具链痛苦与方向排斥）。
- README 与技术报告随 checkpoint 更新，任何一周中断都保留可复现的实现状态。
- 最终形态：能独立推导 online softmax 与 block-table mapping；limitations 含 vAttention 反方证据；能解释 Triton split-KV 与 CUDA single-pass port 的范围差异。
