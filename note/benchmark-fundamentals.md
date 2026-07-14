# Benchmark 基础：怎样得到可信的性能数字

## 1. Benchmark 与 correctness test 的职责不同

Correctness test（正确性测试）回答：

```text
相同输入下，kernel 输出是否与 FP32 reference 对齐？
```

Benchmark（基准测试）回答：

```text
在明确的 shape、dtype、硬件和测量口径下，这条执行路径有多快、是否稳定、如何扩展？
```

性能更快不能证明结果正确。反过来，一次测试通过也不能证明性能数字可信。因此本项目的顺序固定为：

```text
correctness guard
-> warmup
-> repeat timing
-> distribution summary
-> CSV
-> plots
-> profiling hypothesis
```

## 2. 为什么 GPU 计时不能直接用普通 Python 时间

CUDA kernel launch（CUDA 内核启动）通常是 asynchronous（异步）的：CPU 把任务放进 CUDA
stream 后可以立刻继续执行。

下面的伪代码主要测到了 CPU 发射开销：

```python
start = time.perf_counter()
kernel()
elapsed = time.perf_counter() - start
```

纯 GPU 路径使用 CUDA events（CUDA 事件）：

```python
start_event.record()
kernel()
end_event.record()
torch.cuda.synchronize()
latency_ms = start_event.elapsed_time(end_event)
```

如果路径包含 Python 循环、`.item()` 或 CPU/GPU 同步，则使用 synchronized wall clock
（同步墙钟时间）：

```python
torch.cuda.synchronize()
start = time.perf_counter()
python_and_cuda_operation()
torch.cuda.synchronize()
latency_ms = now() - start
```

两种数字的含义不同，不能只看数值大小直接比较。因此 CSV 同时记录
`measurement_scope` 和 `timing_method`。

## 3. Warmup、repeat、p50 和 p95

Warmup（预热）用于排除：

- Triton JIT compilation（即时编译）；
- CUDA context 初始化；
- allocator（分配器）和缓存冷启动；
- GPU 从低功耗状态升频的早期波动。

Repeat（重复测量）不是简单地“循环很多次再除一下”，而是保留每次样本，观察分布。

设延迟样本排序后为：

$$
t_{(1)} \le t_{(2)} \le \cdots \le t_{(N)}
$$

- `p50`：中位数，描述典型性能，对少量异常值不敏感；
- `p95`：95% 样本不超过该值，描述尾延迟和运行抖动；
- `min`：接近理想状态，但容易过度乐观；
- `mean`：容易被少量长尾样本拉高。

## 4. Decode attention 的解析读取量

当前 MHA、FP16 实现中，一次 decode step 对 K/V 的主要理论读取量为：

$$
\text{KV bytes}
= \left(\sum_b S_b\right)
\times 2
\times H_{kv}
\times D
\times \text{dtype bytes}
$$

其中 `2` 表示 K 和 V。然后用实测延迟计算 effective bandwidth（有效带宽）：

$$
\text{effective bandwidth}
= \frac{\text{KV bytes}}{\text{latency}}
$$

这不是完整的硬件流量：它暂时忽略 Q、output、block table、cache-line transaction 和重复读取。
它的价值是提供统一的解析模型，用于比较不同 context length 和 kernel 版本。

## 5. 当前 benchmark 的公平性边界

固定条件：

```text
q_len:          1
layout:         MHA
head_dim:       128
dtype:          FP16
context_lens:   batch 内统一长度
Triton output:  预分配 FP32 tensor
```

计时区不包含：

- 随机输入构造；
- dense-to-paged cache packing；
- block-table 生成；
- correctness reference；
- Triton 首次 JIT 编译；
- Triton public wrapper 的输入验证。

`PyTorch dense SDPA` 会由框架分配输出，而 Triton 使用预分配输出。因此 CSV 明确记录 scope；
后续可以另加统一 output-allocation 口径，但当前主目标是研究 kernel 本身。

## 6. 第一轮图表应该回答什么

1. `latency vs context length`：延迟是否随 KV 读取量近似线性增长？
2. `effective bandwidth vs context length`：长 context 是否更接近稳定带宽平台？
3. `latency vs batch size`：增加 `(batch, head)` programs 后，GPU 利用率是否改善？
4. `paged / dense ratio`：block-table 间接寻址和离散布局付出了多少代价？

先收集证据，再决定调 `block_t`、`num_warps`，还是实现 split-KV。

## 7. Program Saturation 与带宽平台

当前 single-pass kernel 的 program 数为：

```text
program_count = batch * num_heads
```

SM 数量只能提供并行上限线索，不能直接推出达到带宽峰值所需的 program 数。一个 program
内部可以包含多个 warps 并持续发出内存请求，显存控制器可能在所有 SM 都有任务之前就已经
饱和。因此需要固定每个 program 的工作量，逐步增加 program 数并观察 effective bandwidth。

进入 bandwidth plateau（带宽平台）后的典型信号是：

```text
工作量增加 2x
延迟接近增加 2x
有效带宽几乎不再增长
```

此时继续增加 split 或 programs 不能突破显存带宽，只会增加排队和合并开销。

## 8. 带宽下界与优化方向

对真正 memory-bound 且必须读取全部 K/V 的 kernel：

$$
\text{minimum latency}
\approx \frac{\text{required bytes}}{\text{peak memory bandwidth}}
$$

如果实测已经接近这个下界，优化优先级应转向：

```text
减少必须读取的 token 或 KV heads
降低 KV dtype
去掉重复/无效 transaction
提高 cache reuse
避免不必要的 intermediate state
```

需要区分 nominal utilization 与真实 DRAM utilization：解析模型只计算 useful KV bytes，
忽略 cache-line 放大、metadata、重复读取和 cache hit。没有 NCU counter 时，只能把它当成统一
比较模型，不能声称得到了精确硬件 DRAM 利用率。
