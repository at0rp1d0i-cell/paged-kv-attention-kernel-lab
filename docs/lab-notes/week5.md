# Week 5 Lab Notes

## 本周目标

实现 Triton split-KV partial/reduce 路径，验证 online-softmax state 的分段合并语义，并判断
context parallelism 在哪些 workload 上值得启用。

## 完成内容

- 实现 `(batch, head, split)` partial kernel，输出 FP32 `m/l/acc`；
- 实现 `(batch, head)` reduce kernel，支持 `split=1/4/8/16`；
- correctness tests 覆盖 variable-length batch、random-order block table、partial tail、empty
  split 和 garbage slots，并与 FP32 dense reference 对齐；
- 增加 program-matched equal-work benchmark，固定 `B*S`、main program 数和每个 program 的
  token 数，隔离 split/reduce 开销；
- 完成 42 个 `(B,S)` shape、210 行的 same-shape canonical sweep，并生成 latency、speedup、
  measured-best dispatch 与 adaptive dispatch 图；
- 实现基于 `batch * heads`、context 和 block size 的 adaptive dispatch，增加 selector 数据契约
  测试与 single/split/保守回退三类 GPU correctness tests；
- 与 FlashInfer 做同进程交错 recheck，并查看其 scheduler 与 decode CUDA source。

Triton Split-KV checkpoint 已完成；下一阶段不再扩展 Triton 教学，先完成 CUDA design sketch。

## 核心结论

split-KV 在算法上没有引入新的 attention 数学：它把 `(batch, head)` 的 context 顺序扫描改为
`(batch, head, split)` 并行扫描，再利用 online-softmax state 的可结合性合并。真正需要设计的
是 dispatch policy：小 batch、长 context 用 split 补足 program-level parallelism；大 batch
已经达到 bandwidth plateau 时保留 single-pass，避免 intermediate state 和 reduce 开销。

不能只用 `target_programs / (B*H)` 机械选择 split。短 context 下每段工作太少，launch/reduce
成本可能超过并行收益；variable-length batch 还会产生负载不均和 empty split。最终 policy 必须
来自 same-shape sweep，并对 `B=16/32` 的回退设置测试边界。

正式 sweep 中，`S=16K` 的 adaptive speedup 随 batch 增加依次为：`B=1: 10.50x`、
`B=2: 5.70x`、`B=4: 2.08x`、`B=8: 1.10x`；`B=16/32` 保留 single-pass。
42 个 adaptive choice 相对 single-pass 均无回退，`B>=16,S>=4K` 也不会错误启用 split。
这说明 split-KV 解决的是小 program-count 下的并行度不足，而不是降低 attention 的总 KV 读取量。

## Benchmark 与 Profiling 收获

equal-work 和 same-shape 回答不同问题：前者解释 kernel 机制，后者才决定真实 dispatch。顺序测量
曾把 single 相对 split 的优势放大到约 `1.47x`；改成确定性交错测量后，同工作量差距收敛到约
`1.06x`，说明动态时钟、热缓存和 provider 顺序足以扭曲微秒级结论。64 MiB useful KV working
set 还会受 L2 复用影响，解析 effective bandwidth 超过 DRAM peak 不代表物理带宽超频。

一次 exploratory profiler recheck 显示，当前 Triton partial kernel 与 FlashInfer decode 主
kernel 的差距大于两边 reduce/merge 的差距。FlashInfer source 进一步确认它会根据 occupancy
估计切分 chunk，并使用 16-byte vectorized load、shared-memory staging 和 `cp.async` pipeline。
这说明 FlashInfer 的优势不只是“选了 split”，还包括主 kernel 的 CUDA 工程优化。Profiler
数字只用于定位结构，正式 latency 仍以 CUDA events 为准。

policy 相对单轮逐点最优的最大损失是 `1.087x`，发生在 `B=8,S=1K`。这里仍选择 single，原因
是短 context 的绝对延迟处于几十微秒量级，跨 run 波动容易改变微小排序；静态 policy 更应保护
明确的收益区，而不是拟合每个测量点。当前策略只对 RTX 5090、FP16、`H=8`、`D=128`、
`block=32` 的 equal-length sweep 有性能证据。

## 方向感受

这一阶段的学习新意较低，核心实现更像在 single-pass 外再增加一层切分与合并。相比继续扩展
split-KV，更希望下一阶段学习 CUDA 中显式的线程映射、vectorized load、shared-memory staging
和 warp reduction。这不是对 GPU kernel 方向失去兴趣，而是学习重点从算法分段转向底层执行。

## Remaining Work

- 完成 `docs/cuda-design-sketch.md` 后，再开始 CUDA/C++ single-pass port。
- 如需把 adaptive dispatch 用于其他 GPU、block size 或 variable-length workload，应在对应环境
  重跑 same-shape sweep，而不是沿用当前阈值。
