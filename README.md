# Paged-KV Decode Attention Kernel Lab

面向 LLM decode inference 的 Paged-KV attention kernel 实验项目。项目从 FP32 reference 出发，实现 Triton single-pass paged decode attention、split-KV partial/reduce kernels 和基于实测证据的 adaptive dispatch，并建立 correctness、benchmark 与 profiling 闭环。

## 项目概览

当前实现聚焦 `q_len=1` 的 decode attention，使用 block table 把逻辑 token 位置映射到离散物理 KV blocks。项目不包含完整 serving engine 或 KV allocator，重点是理解并验证 paged layout、online softmax、GPU program-level parallelism 和显存带宽之间的关系。

| 组件 | 状态 | 主要能力 |
| --- | --- | --- |
| PyTorch reference | 已完成 | FP32 dense/paged ground truth、variable-length batch |
| Block-table generator | 已完成 | 随机物理块顺序、碎片化映射、garbage slots |
| Triton single-pass | 已完成 | FP16、MHA、`head_dim=128`、paged layout、tail mask |
| Triton split-KV | 已完成 | `split=1/4/8/16`、partial `m/l/acc`、reduce kernel |
| Adaptive dispatch | 已完成 | 根据 program saturation 与 context length 选择 single/split path |
| Benchmark/profiling | 已完成 | CUDA events、p50/p95、CSV、图表、解析有效带宽 |
| CUDA/C++ extension | 待实现 | 限定范围的 single-pass paged decode port |

## 关键结果

正式结果来自 NVIDIA GeForce RTX 5090、FP16、`H=8`、`D=128`、`block_size=32`。详细测量口径、CSV 和限制见 [Benchmark Results](docs/benchmark-results.md) 与 [Profiling Report](docs/profiling-report.md)。

| Shape | Adaptive path | p50 latency | Speedup vs single-pass |
| --- | --- | ---: | ---: |
| `B=1,S=16K` | `split=16` | `0.0259 ms` | `10.50x` |
| `B=2,S=16K` | `split=16` | `0.0548 ms` | `5.70x` |
| `B=4,S=16K` | `split=4` | `0.1610 ms` | `2.08x` |
| `B=8,S=16K` | `split=4` | `0.3253 ms` | `1.10x` |
| `B=16/32,S=16K` | single-pass | `0.6385/1.2765 ms` | `1.00x` |

- Canonical same-shape sweep 覆盖 42 个 `(batch, context)` shape，adaptive choice 相对 single-pass 均无回退。
- `B=1,S=16K` 的 single-pass Paged Triton 只有 8 个 programs，有效带宽约 `246 GB/s`；`B=16` 增加到 128 个 programs 后达到约 `1679 GB/s`，接近项目采用的 `1792 GB/s` 标称峰值假设。
- FlashInfer 在 `B=1,S=16K` 的 p50 为 `0.0237 ms`，验证了 context parallelism 对小 batch、长 context 的价值；到 `B=16`，Paged Triton、FlashInfer 与 PyTorch SDPA 收敛到相近的带宽平台。

![Adaptive split-KV speedup](benchmarks/results/split_kv_same_shape_speedup_by_batch.png)

![Program saturation](benchmarks/results/decode_attention_program_saturation_batch_scaling.png)

## 问题痛点

### Decode 阶段的带宽压力

在 autoregressive decode 中，每一步只有一个 query token，却需要读取完整历史 K/V：

```text
softmax(Q K^T / sqrt(d)) V
```

Q 的数据量很小，K/V 随 context length 线性增长。长上下文下，kernel 通常更受 KV memory traffic 约束，而不是受矩阵计算吞吐约束。

### KV Cache 的动态内存管理

在线请求的 prompt 长度、decode 长度和结束时间不同。为每个请求预留连续 KV cache 会产生显存浪费和碎片。Paged KV cache 把逻辑序列拆成固定大小的物理块，通过 block table 管理映射，使分配与回收更灵活。

### 分页布局的访存代价

Paged layout 引入 block-table lookup、物理块跳转、末块 mask 和潜在的非连续访问。系统层需要灵活的内存管理，kernel 层则希望连续、合并且规则的访存，两者之间存在直接取舍。

### 小批量长上下文的并行度

Single-pass kernel 为每个 `(batch, head)` 启动一个 Triton program：

```text
program_count = batch_size * num_heads
```

`B=1,H=8` 时只有 8 个 programs，每个 program 顺序扫描长 context，无法充分占用 GPU。Split-KV 把 context 划分为多个并行片段，再合并 online-softmax state，从而补足 program-level parallelism。

## 数据布局

核心接口采用以下 tensor contract：

```python
def paged_decode_attention(
    q,              # [B, Hq, D]
    k_cache,        # [num_blocks, block_size, Hkv, D]
    v_cache,        # [num_blocks, block_size, Hkv, D]
    block_tables,   # [B, max_blocks_per_sequence]
    context_lens,   # [B]
    *,
    block_size: int,
    scale: float,
):
    # output: [B, Hq, D]
```

当前 Triton 路径限定为 MHA，因此 `Hq == Hkv`。逻辑 token 到物理 cache slot 的映射为：

```text
logical_block = token_position // block_size
slot          = token_position % block_size
physical_block = block_tables[batch, logical_block]
```

测试数据会把未使用 block 和 slot 填为大幅 garbage values；任何错误的间接寻址或 tail mask 都会显著破坏输出。

## 核心实现

### FP32 Reference

`src/paged_kv_attention/reference.py` 提供 dense attention、block-wise online-softmax reference 和 paged reference。Correctness 测试始终以 FP32 dense result 为 ground truth，不以另一个优化 kernel 作为真值。

### Single-Pass Triton

每个 Triton program 负责一个 `[D]` 输出向量，在 context tile 上维护 online-softmax state：

```text
m: running maximum
l: running exponential sum
acc: running weighted-value accumulator
```

新 tile 到达时，旧 state 会根据新的全局 maximum 重新缩放，避免物化完整 score/probability tensor，同时保持数值稳定。

### Split-KV Triton

Split-KV partial kernel 使用 `(batch, head, split)` grid，每个 split 独立输出 FP32 `m/l/acc`。Reduce kernel 利用 online-softmax state 的可结合性合并所有 partial states，得到最终 `[B,H,D]` 输出。

Split-KV 不减少 KV 总读取量。它解决的是小 program count 下的并行度不足，同时引入 intermediate state、第二次 kernel launch 和 reduce 成本。

### Adaptive Dispatch

静态 dispatch policy 来自 same-shape sweep，而不是只按目标 program 数推导。策略同时考虑：

- `batch_size * num_heads` 的基础 program 数；
- context length 是否足以摊薄 launch/reduce 成本；
- 大 batch 是否已经进入 memory-bandwidth plateau；
- 当前是否为已校准的 `block_size=32`。

未测量的 block size 保守回退 single-pass。Variable-length batch 使用最大 context length 选路，已验证 correctness，但尚未形成独立性能结论。

## 正确性验证

Correctness suite 覆盖：

- batch 1 与 multi-batch；
- context 小于、等于和跨越一个 block；
- 最后一个 block 未填满；
- variable-length batch；
- non-contiguous、random-order block table；
- 未使用 block/slot 的 garbage values；
- split `1/4/8/16`、partial tail 与 empty split；
- adaptive single/split/fallback paths；
- FP16 Triton output 与 FP32 dense reference 的 tolerance 对齐。

当前验证结果：

```text
CPU tests: 53 passed
GPU tests: 22 passed
Ruff:      passed
```

运行测试：

```bash
bash scripts/run_tests.sh
uv run pytest -m gpu
uv run ruff check .
```

## 性能评估

### 测量方法

- GPU latency 使用 CUDA events；
- 每个正式 shape 包含 warmup、repeat、p50 与 p95；
- 输入生成、cache packing、correctness guard 和 JIT 编译不计入 kernel latency；
- 环境事实与 clock state 写入 CSV；
- 图表只从 canonical CSV 生成；
- effective bandwidth 使用解析 useful KV bytes 除以实测 latency。

解析 KV bytes：

```text
sum(context_lens) * 2(K+V) * num_kv_heads * head_dim * dtype_size
```

该指标不是 NCU DRAM transaction counter。重复读取相同 tensor 时，L2 cache 复用可能使解析有效带宽超过标称 DRAM peak。

### 对照实现

- PyTorch dense SDPA；
- PyTorch paged reference；
- Triton dense single-pass；
- Triton paged single-pass；
- Triton paged split-KV；
- FlashInfer paged decode wrapper。

### 性能证据

![FlashInfer latency comparison](benchmarks/results/decode_attention_flashinfer_latency_p50_by_batch.png)

![Adaptive dispatch map](benchmarks/results/split_kv_same_shape_adaptive_dispatch_map.png)

完整结果、测量契约和复现命令见 [docs/benchmark-results.md](docs/benchmark-results.md)。Profiling 结论和工具限制见 [docs/profiling-report.md](docs/profiling-report.md)。

## 环境配置

项目使用 `uv` 管理 Python 环境。当前已验证环境为 Python 3.12、PyTorch `2.13.0+cu130`、Triton `3.7.1` 和 RTX 5090。完整版本与工具链事实见 [docs/env-notes.md](docs/env-notes.md)。

```bash
cd /root/paged-kv-attention-kernel-lab
python -m pip install -U uv
UV_HTTP_TIMEOUT=600 uv sync --locked --group dev
bash scripts/check_env.sh
bash scripts/run_tests.sh
uv run python scripts/gpu_smoke.py
```

FlashInfer baseline 使用可选的 `baseline` 依赖组，安装到项目现有的 `.venv` 中：

```bash
UV_HTTP_TIMEOUT=600 uv sync --locked --group baseline
uv run --group baseline python scripts/flashinfer_smoke.py
```

当前容器中的 Nsight Compute 可以启动，但 GPU performance counters 被 `ERR_NVGPUCTRPERM` 阻止。项目使用 CUDA events、`torch.profiler` timeline 和 analytical bandwidth model 作为 fallback。

## Benchmark 复现

快速 benchmark：

```bash
bash scripts/run_benchmarks.sh \
  --batches 1 \
  --contexts 128,512 \
  --block-sizes 16 \
  --warmup 5 \
  --repeat 20
```

Adaptive split-KV sweep：

```bash
uv run python scripts/run_split_kv_benchmarks.py \
  --batches 1,2,4,8,16,32 \
  --contexts 512,1024,2048,4096,8192,16384,32768 \
  --splits 1,4,8,16 \
  --warmup 50 \
  --repeat 300 \
  --peak-bandwidth-gbps 1792
```

Profiler capture：

```bash
uv run python scripts/profile_decode_attention.py \
  --batch-size 1 \
  --context-len 16384 \
  --block-size 32
```

## 仓库结构

```text
src/paged_kv_attention/       reference、layout、Triton kernels、benchmark helpers
tests/                        CPU contract tests 与 GPU numerical tests
scripts/                      smoke、benchmark、plot、profiling 与环境脚本
benchmarks/results/           canonical CSV 与由 CSV 生成的图表
docs/                         环境、benchmark、profiling、roadmap 辅助文档
docs/lab-notes/               按阶段记录的实验过程与复盘
note/                         概念推导与实现学习笔记
```

## 限制与后续工作

当前限制：

- Triton kernel 仅支持 FP16、MHA、`head_dim=128`；
- adaptive policy 只在 RTX 5090、FP16、`H=8`、`D=128`、`block_size=32` 的 equal-length shapes 上校准；
- variable-length workload 已验证 correctness，未单独校准 dispatch 性能；
- NCU counters 不可用，occupancy、register usage 和实际 DRAM transactions 缺少直接硬件计数；
- benchmark 重复读取相同 K/V，部分 shape 会受 L2 cache 复用影响；
- 项目不包含 serving scheduler、KV allocator、prefix cache 或 continuous batching runtime。

后续主线：

1. 完成 CUDA design sketch，明确 thread-block mapping、shared-memory staging、online-softmax reduction 与 vectorized loads。
2. 实现限定范围的 CUDA/C++ single-pass paged decode extension，并复用现有 correctness suite。
3. 对比 Triton 与 CUDA 的接口、延迟、有效带宽和工程复杂度。
4. 整理最终技术报告，并补充 vAttention 对 PagedAttention 间接寻址成本的反方证据。

GQA/MQA、BF16、`head_dim=64` 和更多 block-size sensitivity 属于后续扩展，不阻塞当前 Triton split-KV checkpoint。

## 项目文档

- [Roadmap](ROADMAP.md)
- [Acceptance Criteria](ACCEPTANCE_CRITERIA.md)
- [Benchmark Results](docs/benchmark-results.md)
- [Profiling Report](docs/profiling-report.md)
- [Environment Notes](docs/env-notes.md)
- [Learning Syllabus](docs/learning-syllabus.md)
- [Reading List](docs/reading-list.md)
