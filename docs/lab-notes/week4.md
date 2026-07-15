# Week 4 Lab Notes

## 本周目标

建立可复现的 decode attention benchmark/profiling 证据链，并从数据决定下一阶段优化方向。

## 最难的问题

最难的不是调用 CUDA events，而是定义 measurement scope（测量口径）。Triton public wrapper
包含 GPU validation reduction，不能直接当成 raw kernel latency；PyTorch paged reference 又包含
Python loop 和 `.item()`，必须用 synchronized wall clock。不同口径如果不写进 CSV，会产生
看似精确但无法比较的数字。

第二个问题是区分 program 数、SM 数和带宽饱和点。RTX 5090 有 170 个 SM，但实验显示
64-128 个 `(batch, head)` programs 已能达到约 87%-94% 的标称峰值带宽，因此“program 数必须
超过 SM 数才会饱和”是错误推断，饱和点必须实测。

## 关键数据

固定 `S=16384,H=8,D=128`，Paged block-32：

```text
B=1:    8 programs, 约 246 GB/s, 13.7%
B=8:   64 programs, 约 1495 GB/s, 83.4%
B=16: 128 programs, 约 1679 GB/s, 93.7%
B=32: 256 programs, 约 1685 GB/s, 94.0%
```

从 128 到 256 programs 后带宽几乎不增长，说明大 batch 已到 bandwidth plateau。小 batch
需要增加 context parallelism，大 batch 应保留 single-pass。

## Profiling 收获

当前 PyTorch 2.13.0 / CUPTI 13.0.85 的 `torch.profiler` 已恢复 CUDA kernel events。
`B=1,S=16384` 的 timeline 显示 Dense Triton、Paged Triton、SDPA split-KV 与 combine 分别约为
`257.3 / 334.6 / 37.6 / 6.5 us`；`B=16` 时分别约为
`628.8 / 637.0 / 641.6 / 4.5 us`。Profiler instrumentation 会扰动微秒级 kernel，正式延迟和
优化判断仍以 CUDA events 为准，timeline 用于确认 kernel selection 与 split/reduce 结构。

FlashInfer baseline 最初受系统 CUDA 12.8 编译器阻塞；进一步定位到它按 `CUDA_HOME` 检查
编译器版本。最终在可选 baseline 依赖组中固定 `cuda-toolkit[nvcc]==13.0.3`，实际 `nvcc`
版本为 13.0.88，并自动补齐 pip toolkit 与传统 toolkit 的目录差异。整个过程没有修改
FlashInfer 源码。

正式 sweep 显示，`B=1,S=16384,block=32` 时 FlashInfer 为 `0.0237 ms`，Paged
Triton 为 `0.2729 ms`，差距约 `11.53x`；`B=4` 差距约 `2.04x`；到 `B=16` 两者均约
`0.64 ms`。这说明 split-KV 主要解决小 batch 长 context 的并行度不足，而大 batch 已接近
带宽平台。

## 方向感受

目前最享受的是从性能数据中发现模式、提出可证伪假设，再设计聚焦实验验证。相比单纯增加
代码量，我更愿意继续做 benchmark、profiling 和由数据驱动的 kernel 优化。因此项目路线
确定为先实现 Triton adaptive split-KV，再串行完成限定范围的 CUDA/C++ single-pass port。

## 下一步

- 实现 split partial `m/l/acc` 接口与 correctness tests。
- 实现 reduce kernel，验证 split 合并公式。
- sweep `split=4/8/16`，对比 partial/reduce 成本和有效带宽收益。
- 用 adaptive dispatch 保护已经带宽饱和的 `batch=16/32`。

## Git Checkpoint

当前 performance checkpoint 建议按三组语义提交：

```text
1. benchmark harness、CUDA-event timing、CSV/plot 与 utility tests
2. canonical results、profiling report、lab note 与 resume checkpoint
3. split-KV -> minimal CUDA port 的路线与验收文档调整
```

提交前保留 FlashInfer correctness、CSV 与关键图表，并在简历中只引用已复现数据。
