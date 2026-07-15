# Profiling Report

## Scope

本报告分析 single-pass dense/paged Triton decode attention、split-KV 收益边界与 adaptive
dispatch。GPU 为 NVIDIA GeForce RTX 5090，PyTorch `2.13.0+cu130`，Triton
`3.7.1`，FP16 input、FP32 accumulation、`H=8`、`D=128`、`S=16384`。

完整 NCU counter collection 因 `ERR_NVGPUCTRPERM` 不可用。当前证据由三部分组成：

```text
CUDA events latency
torch.profiler CUDA kernel timeline
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

## Profiler Status

当前 `torch.profiler` 可以记录 CUDA kernel events。`B=1,S=16384` 时：

```text
Dense Triton:        257.3 us
Paged Triton:        334.6 us
SDPA split-KV:        37.6 us
SDPA combine:          6.5 us
```

`B=16,S=16384` 时：

```text
Dense Triton:        628.8 us
Paged Triton:        637.0 us
SDPA split-KV:       641.6 us
SDPA combine:          4.5 us
```

Profiler instrumentation 会扰动微秒级 kernel，因此正式 latency 仍以 CUDA events 为准；
timeline 用于确认 SDPA 的 split-KV + combine 结构和相对成本。

## Memory-Bound Evidence

解析 KV bytes：

```text
sum(context_lens) * 2(K+V) * H_kv * D * dtype_size
```

Paged Triton block-32、`S=16384`：

| Batch | Programs | p50 | Effective bandwidth | Nominal peak utilization |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.273 ms | 246 GB/s | 13.7% |
| 8 | 64 | 0.359 ms | 1495 GB/s | 83.4% |
| 16 | 128 | 0.640 ms | 1679 GB/s | 93.7% |
| 32 | 256 | 1.274 ms | 1685 GB/s | 94.0% |

从 `128` 增加到 `256` programs 后，工作量和延迟近似翻倍，但有效带宽只增加约 `0.5%`，
说明大 batch 已进入 memory-bandwidth plateau。此时继续增加 context split 只会增加 partial
state 和 reduce 开销。

## Paged Addressing Cost

`B=4,S=16384` 时：

```text
Dense Triton:       0.3308 ms
Paged Triton b16:   0.3774 ms  (+14.1%)
Paged Triton b32:   0.3385 ms  (+2.3%)
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

`B=1,H=8` 只有 8 个 programs，有效带宽约为标称峰值的 13.7%。`B=8` 增加到 64 个 programs
后达到约 83.4%，`B=16` 的 128 个 programs 接近平台。这个曲线说明问题是 program-level
parallelism 不足，而不是单纯需要继续增大 `block_t`。

Split-KV 应优先测试：

```text
split=4  -> 32 programs
split=8  -> 64 programs
split=16 -> 128 programs
```

same-shape sweep 验证了这个方向，同时给出了停止 split 的边界：`S=16384` 时，adaptive path
在 `B=1/2/4/8` 分别获得 `10.50x / 5.70x / 2.08x / 1.10x`，到 `B=16/32` 选择
single-pass。`B>=16,S>=4096` 的保护规则避免在已接近带宽平台时增加 partial state、第二次
kernel launch 和 reduce。

## Limitations

- GPU clocks 未锁定，p95 存在偶发波动，主要结论使用长 context p50。
- effective bandwidth 使用解析 useful KV bytes，不等于实际 DRAM transaction bytes。
- benchmark 重复读取相同 tensor，小工作集可能受 L2 cache 影响。
- NCU 因权限不可用，无法直接报告 achieved occupancy、DRAM throughput 和 register usage。
- `torch.profiler` 会扰动微秒级 kernel，只用于 timeline 与 kernel-selection 证据。
- adaptive policy 只在 RTX 5090、FP16、`H=8`、`D=128`、`block_size=32` 的 equal-length
  sweep 上校准；其他 block size 保守回退 single，variable-length 性能仍需单独测量。

## Optimization Decision

Triton partial/reduce split-KV 已通过 correctness，equal-work analysis 解释了约 `1.06x` 的
partial/reduce 固定成本，same-shape sweep 则确定了 context threshold 与 split 数。adaptive
dispatch 只在实测可获益区域启用 split，并在 canonical 42 个 shape 上保持相对 single-pass
无回退。这个 profiling checkpoint 已闭环，下一步转入 CUDA single-pass design sketch。
