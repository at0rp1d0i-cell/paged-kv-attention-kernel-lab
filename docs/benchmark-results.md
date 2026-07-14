# Benchmark Results

## Status

第一轮主 sweep 已完成。原始结果位于：

```text
benchmarks/results/decode_attention_main.csv
```

配置：`batch=1/4/16`、`context=128/512/2048/8192/16384`、`block_size=16/32`、
`num_heads=8`、`head_dim=128`、FP16、`warmup=50`、`repeat=300`，共 60 行。

CSV 使用 `1792 GB/s` 作为 RTX 5090 nominal peak memory bandwidth（标称峰值显存带宽）假设，
同时记录 effective bandwidth 和 nominal utilization。该比例是解析模型，不是 NCU DRAM counter。

主要图表：

- [按 batch 比较 p50 latency](../benchmarks/results/decode_attention_main_latency_p50_by_batch.png)
- [按 batch 比较 effective bandwidth](../benchmarks/results/decode_attention_main_bandwidth_by_batch.png)
- [按 batch 比较 nominal peak utilization](../benchmarks/results/decode_attention_main_bandwidth_utilization_by_batch.png)
- [Paged/Dense Triton latency ratio](../benchmarks/results/decode_attention_main_paged_dense_ratio.png)
- [长 context 的 batch scaling](../benchmarks/results/decode_attention_main_batch_scaling.png)

## Reproduction

先验证 correctness（正确性）：

```bash
bash scripts/run_tests.sh
uv run pytest -q tests/test_triton_decode.py
```

运行一个快速 smoke benchmark（冒烟基准测试）：

```bash
bash scripts/run_benchmarks.sh \
  --batches 1 \
  --contexts 128,512 \
  --block-sizes 16 \
  --warmup 5 \
  --repeat 20
```

运行初始主 sweep：

```bash
bash scripts/run_benchmarks.sh \
  --batches 1,4,16 \
  --contexts 128,512,2048,8192,16384 \
  --block-sizes 16,32 \
  --warmup 50 \
  --repeat 300 \
  --peak-bandwidth-gbps 1792
```

从 CSV 生成静态图表：

```bash
uv sync --locked --group plot
uv run --group plot python scripts/plot_benchmarks.py benchmarks/results/<result>.csv
```

## Measurement Contract

- Triton kernel 与 PyTorch dense SDPA 使用 CUDA events 计时。
- Python paged reference 使用 synchronized wall clock，包含 Python 循环和同步成本。
- Python paged reference 默认不加入主 sweep；显式选择后使用独立的较小 `--reference-repeat`。
- 输入生成、paged cache packing、correctness guard 和 Triton JIT 不在计时区内。
- CSV 是 source of truth（唯一数据源），图表只从 CSV 生成。
- 当前 clock state 为 `recorded_not_locked`，每行记录运行时 graphics/memory clock。

## Initial Hypotheses

- 长 context 下延迟主要受 K/V memory traffic（内存流量）约束。
- `batch=1` 时只有 `num_heads` 个 Triton programs，长 context 顺序扫描可能导致 occupancy 不足。
- 增加 batch 后并行 program 数增加，有效带宽应先提高，再逐渐饱和。
- Paged Triton 相对 dense Triton 会承担 block-table lookup 和更离散的内存访问成本。

这些是假设，不是结论。正式图表生成后再逐项接受或否定。

## First Results

长 context 的 p50 latency（ms）：

| Batch | Context | Dense Triton | Paged Triton 16 | Paged Triton 32 | Dense SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16384 | 0.241 | 0.249 | 0.253 | 0.040 |
| 4 | 16384 | 0.290 | 0.348 | 0.316 | 0.173 |
| 16 | 16384 | 0.641 | 0.662 | 0.638 | 0.648 |

`context=16384` 时，Paged Triton 的 effective bandwidth 从 `batch=1` 的约
`265-269 GB/s`（约标称峰值的 `15%`）提升到 `batch=16` 的约 `1622-1684 GB/s`
（约 `91%-94%`）。同一 kernel 随着
`(batch, head)` programs 增加获得约 6 倍有效带宽，支持“小 batch 下 program 数不足、
GPU 未被充分利用”的假设。

Paged Triton 相对 Dense Triton 的长 context 开销并不恒定：

- `batch=1`：约 `2.5%-4.2%`；
- `batch=4`：约 `8.1%-19.8%`；
- `batch=16, block=32`：接近持平。

这说明 paging overhead（分页开销）会与并行度、block size 和内存访问状态共同作用，不能用
单个固定百分比概括。

## Tail-Latency Recheck

主 sweep 中两个最大的 `p95/p50` 异常点被单独用 `warmup=50`、`repeat=300` 复测：

```text
B=1, S=512,  block=32: 2.05x -> 1.17x
B=4, S=2048, block=32: 1.34x -> 1.16x
```

因此原异常更像未锁时钟环境中的偶发抖动，不作为 kernel 固有尾延迟结论。正式报告仍需明确
标注当前 `clock_state=recorded_not_locked`。

两轮完整 sweep 的长 context Triton p50 结果高度一致：大多数点的相对差异低于 `0.3%`，
中位差约 `0.04%`。`PyTorch dense SDPA` 的 `batch=1, context=8192` 出现一次约 `32%`
的 run-to-run 差异，因此该单点不用于推导稳定结论。

## Program Saturation

固定 `context=16384`、`H=8` 后，Paged Triton block-32 的 program saturation 结果为：

| Batch | Programs (`B*H`) | Effective bandwidth | Nominal utilization |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 265 GB/s | 15% |
| 2 | 16 | 425 GB/s | 24% |
| 4 | 32 | 852 GB/s | 48% |
| 8 | 64 | 1562 GB/s | 87% |
| 16 | 128 | 1676-1684 GB/s | 94% |
| 32 | 256 | 1693 GB/s | 94.5% |

`128 -> 256` programs 后工作量与延迟近似同时翻倍，而带宽几乎不再提升，因此 split-KV 的
首批候选应为 `split=4/8/16`，不默认使用 `split=32`。

## Baseline Status

PyTorch paged reference 仅做教学下界：`B=1,S=128,H=8,block=16` 的 synchronized wall
time 为 `7.925 ms`。它包含 Python token loop、`.item()` 与同步，不与 raw GPU kernel 做
生产级公平比较。原始行位于 `benchmarks/results/pytorch_paged_reference_smoke.csv`。

FlashInfer `0.6.14` 可以安装并导入，但当前固定环境无法执行 paged decode：

```text
GPU:                NVIDIA GeForce RTX 5090 / SM 12.0
PyTorch:            2.8.0+cu128
FlashInfer warning: SM 12.x requires CUDA >= 12.9
plan() error:       RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

复现：

```bash
uv sync --locked --group baseline
uv run --group baseline python scripts/flashinfer_smoke.py
```

这是 FlashInfer JIT 在 CUDA 12.8 下无法识别/支持 SM12 的兼容性阻塞，不是 tensor layout
适配错误。当前不升级已通过 Triton correctness 的 PyTorch/CUDA stack，也不声称完成了
FlashInfer 定量对比。

## Next Measurement

当前证据已经足以进入 Triton split-KV：为 `batch=1/2`、长 context 增加 context parallelism，
对比 partial/reduce 成本与有效带宽收益，并通过 adaptive dispatch 保护带宽已饱和的大 batch。
