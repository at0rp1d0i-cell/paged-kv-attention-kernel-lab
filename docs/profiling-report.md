# Profiling Report

## Scope

本报告分析当前 single-pass dense/paged Triton decode attention，并为 Triton split-KV 建立
profiling baseline。GPU 为 NVIDIA GeForce RTX 5090，PyTorch `2.9.1+cu130`，Triton
`3.5.1`，FP16 input、FP32 accumulation、`H=8`、`D=128`、`S=16384`。

完整 NCU counter collection 因 `ERR_NVGPUCTRPERM` 不可用。当前证据由两部分组成：

```text
CUDA events latency
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

当前 PyTorch 2.9.1 / CUPTI 13.0.48 的 `torch.profiler` 运行只记录到 CPU events，没有生成
CUDA kernel events。隔离环境中的 PyTorch 2.13.0 / CUPTI 13.0.85 可以正常记录 `kernel`、
`cuda_runtime` 和 CUDA time，因此问题位于当前 Kineto/CUPTI stack，而不是 RTX 5090、权限或
脚本。本报告不从 profiler 推导 kernel latency 或 kernel-selection 结论；正式延迟来自
CUDA-event benchmark，split-KV 方向由 program saturation 与 FlashInfer 对照数据支持。

## Memory-Bound Evidence

解析 KV bytes：

```text
sum(context_lens) * 2(K+V) * H_kv * D * dtype_size
```

Paged Triton block-32、`S=16384`：

| Batch | Programs | p50 | Effective bandwidth | Nominal peak utilization |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.243 ms | 276 GB/s | 15.4% |
| 8 | 64 | 0.338 ms | 1590 GB/s | 88.8% |
| 16 | 128 | 0.639 ms | 1681 GB/s | 93.8% |
| 32 | 256 | 1.272 ms | 1689 GB/s | 94.2% |

从 `128` 增加到 `256` programs 后，工作量和延迟近似翻倍，但有效带宽只增加约 `0.5%`，
说明大 batch 已进入 memory-bandwidth plateau。此时继续增加 context split 只会增加 partial
state 和 reduce 开销。

## Paged Addressing Cost

`B=4,S=16384` 时：

```text
Dense Triton:       0.3245 ms
Paged Triton b16:   0.3747 ms  (+15.5%)
Paged Triton b32:   0.3068 ms  (-5.5%)
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

`B=1,H=8` 只有 8 个 programs，有效带宽约为标称峰值的 15.4%。`B=8` 增加到 64 个 programs
后达到约 88.8%，`B=16` 的 128 个 programs 接近平台。这个曲线说明问题是 program-level
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
- NCU 因权限不可用，无法直接报告 achieved occupancy、DRAM throughput 和 register usage。
- 当前 PyTorch 2.9.1 / CUPTI 13.0.48 trace 未记录 CUDA kernel events；升级到已验证可用的
  profiler stack 会改变主环境，必须与全量 correctness 和 benchmark re-baseline 一起进行。

## Optimization Decision

下一阶段实现 Triton partial/reduce split-KV，并用 adaptive dispatch 只覆盖小 batch、长 context。
大 batch single-pass 已接近带宽物理下界，不应无条件增加 split 与 reduce。
