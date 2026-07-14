# Profiling Report

## Scope

本报告分析当前 single-pass dense/paged Triton decode attention，并为 Triton split-KV 建立
profiling baseline。GPU 为 NVIDIA GeForce RTX 5090，PyTorch `2.8.0+cu128`，Triton
`3.4.0`，FP16 input、FP32 accumulation、`H=8`、`D=128`、`S=16384`。

完整 NCU counter collection 因 `ERR_NVGPUCTRPERM` 不可用。当前证据由三部分组成：

```text
CUDA events latency
torch.profiler operator/kernel timeline
analytical KV bandwidth model
```

## Reproduction

```bash
uv run python scripts/profile_decode_attention.py \
  --batch-size 1 --context-len 16384 --block-size 32

uv run python scripts/profile_decode_attention.py \
  --batch-size 16 --context-len 16384 --block-size 32
```

文本表和 Chrome trace 写入本地 `profiles/`，原始 trace 不进入 Git。

## Kernel Timeline

`B=1,S=16384` 的 profiler CUDA average：

| Path | CUDA average |
| --- | ---: |
| Dense Triton | 240 us |
| Paged Triton | 313 us |
| PyTorch dense SDPA | 45 us |

`B=16,S=16384`：

| Path | CUDA average |
| --- | ---: |
| Dense Triton | 627 us |
| Paged Triton | 637 us |
| PyTorch dense SDPA | 644 us |

Profiler instrumentation 会扰动微秒级 kernel，因此正式延迟取自 CUDA-event benchmark；这里
主要使用 kernel 名称和相对结构做归因。

最重要的 timeline 证据是 PyTorch SDPA 在 `q_len=1`、长 context 下调用：

```text
flash_fwd_splitkv_kernel
flash_fwd_splitkv_combine_kernel
```

`B=1` 时 split kernel 约 `37 us`，combine 约 `6 us`。成熟 baseline 已通过 context split
增加并行度，这与本项目根据 saturation experiment 得出的 Triton split-KV 方向一致。

## Memory-Bound Evidence

解析 KV bytes：

```text
sum(context_lens) * 2(K+V) * H_kv * D * dtype_size
```

Paged Triton block-32、`S=16384`：

| Batch | Programs | p50 | Effective bandwidth | Nominal peak utilization |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.253 ms | 265 GB/s | 15% |
| 8 | 64 | 0.344 ms | 1562 GB/s | 87% |
| 16 | 128 | 0.638 ms | 1684 GB/s | 94% |
| 32 | 256 | 1.268 ms | 1693 GB/s | 94.5% |

从 `128` 增加到 `256` programs 后，工作量和延迟近似翻倍，但有效带宽只增加约 `0.5%`，
说明大 batch 已进入 memory-bandwidth plateau。此时继续增加 context split 只会增加 partial
state 和 reduce 开销。

## Paged Addressing Cost

`B=4,S=16384` 时：

```text
Dense Triton:       0.290 ms
Paged Triton b16:   0.348 ms  (+20%)
Paged Triton b32:   0.316 ms  (+9%)
```

`B=16,S=16384` 时 Paged block-32 与 Dense 基本持平。说明 block-table lookup 和物理块跳转
的成本会被 workload 并行度与 latency hiding 改变，不能用单一固定百分比概括。Block-32
跨越更少的 physical-block boundaries，但当前 kernel 仍为每个 token lane 构造 table 地址，
因此不能解释为 lookup 指令简单减半。

## Online Softmax Cost

当前 score、online softmax 和 V accumulation 融合在同一个 Triton kernel 中，
`torch.profiler` 无法拆分它们的内部耗时。可以确认的是：single-pass kernel 不物化完整 scores
或 probabilities，且每个有效 K/V 元素只需参与一次 streaming pass。没有 NCU/source-level
counter 时，不对 online-softmax 指令占比给出伪精确百分比。

## Small-Batch Occupancy Problem

当前 Triton grid 为：

```text
program_count = batch * num_heads
```

`B=1,H=8` 只有 8 个 programs，有效带宽约为标称峰值的 15%。`B=8` 增加到 64 个 programs
后达到约 87%，`B=16` 的 128 个 programs 接近平台。这个曲线说明问题是 program-level
parallelism 不足，而不是单纯需要继续增大 `block_t`。

Split-KV 应优先测试：

```text
split=4  -> 32 programs
split=8  -> 64 programs
split=16 -> 128 programs
```

## Limitations

- GPU clocks 未锁定，p95 存在偶发波动，主要结论使用长 context p50。
- effective bandwidth 使用解析 useful KV bytes，不等于实际 DRAM transaction bytes。
- benchmark 重复读取相同 tensor，小工作集可能受 L2 cache 影响。
- `torch.profiler` 会扰动微秒级 kernel，只用于 timeline 和 kernel-selection 证据。
- NCU 因权限不可用，无法直接报告 achieved occupancy、DRAM throughput 和 register usage。
- FlashInfer 0.6.14 在 RTX 5090 + PyTorch CUDA 12.8 下无法通过 SM12 JIT capability check。

## Optimization Decision

下一阶段实现 Triton partial/reduce split-KV，并用 adaptive dispatch 只覆盖小 batch、长 context。
大 batch single-pass 已接近带宽物理下界，不应无条件增加 split 与 reduce。
