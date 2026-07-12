# Triton Dense Decode Attention 学习笔记

这份笔记记录连续 dense KV layout（稠密 KV 布局）上的第一个 Triton decode
attention（Triton 解码注意力）kernel（内核）。online softmax（在线 softmax）的完整
数学推导见 `note/online-softmax.md`；本文重点记录 program mapping（程序映射）、
pointer arithmetic（指针运算）、mask（掩码）和 shape（形状）。

## 1. 接口与当前限制

```text
q:             [B, H, D]
k / v:         [B, S, H, D]
context_lens:  [B]
out:           [B, H, D]
```

当前范围：

```text
q_len = 1
MHA
head_dim = 128
FP16 CUDA contiguous input
FP32 accumulation and output
context_len > 0
```

正确性标准是与 FP32 `dense_decode_attention` 对齐。

## 2. 一个 Program 负责什么

kernel launch（内核启动）使用：

```python
grid = (batch_size, num_heads)
```

每个 program 固定一个 `(batch_idx, head_idx)`：

```python
batch_idx = tl.program_id(axis=0)
head_idx = tl.program_id(axis=1)
```

它负责完整输出：

```text
out[batch_idx, head_idx, :]  # [D]
```

`D` 不放入 grid，而是在 program 内向量化处理：

```python
offs_d = tl.arange(0, block_d)  # [D]
```

状态 shape：

```text
q_vec:         [D]
running_max:   scalar
running_sum:   scalar
accumulator:   [D]
```

softmax 沿 token 维度归约，所以 max 和 exp sum 是标量；最终输出仍有 `D` 维，所以
accumulator 是 `[D]`。

## 3. 从逻辑索引到数据

连续 K/V layout 是 `[B, S, H, D]`。元素 `k[b,t,h,d]` 的线性偏移为：

$$
\operatorname{offset}(b,t,h,d)
= (((bS+t)H+h)D+d)
$$

逐层看：

```text
b * S + t                     选择 batch 和 token
(b * S + t) * H + h           进入对应 head
((b * S + t) * H + h) * D+d   进入 head 内的 dim
```

当前 tile 的逻辑索引是：

```python
offs_t = start + tl.arange(0, block_t)  # [T]
offs_d = tl.arange(0, block_d)          # [D]
```

`offs_t/offs_d` 首先是 token/dim 逻辑下标。增加单例维度后发生 broadcasting（广播）：

```text
offs_t[:, None]: [T, 1]
offs_d[None, :]: [1, D]

[T, 1] 与 [1, D] -> [T, D]
```

```python
offs_kv = (
    ((batch_idx * max_context_len + offs_t[:, None])
    * num_heads + head_idx)
    * head_dim + offs_d[None, :]
)  # [T, D]
```

三个变量的语义：

```text
offs_kv:  相对 tensor 起点的线性偏移 [T, D]
k_ptrs:   当前 tile 的一组内存地址     [T, D]
k_tile:   从这些地址加载出的 K 数据    [T, D]
```

```python
k_ptrs = k_ptr + offs_kv
k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
```

`k_ptr` 是整个 K tensor 的基础指针，`k_ptrs` 是当前 program 要读取的一组地址。

## 4. Mask 的两层边界

```python
valid_t = offs_t < context_len  # [T]
valid_d = offs_d < head_dim     # [D]
kv_mask = valid_t[:, None] & valid_d[None, :]  # [T, D]
```

`valid_t` 防止超过当前序列的有效 context。超出的 token 即使仍在 `[B,S,H,D]` 的分配
范围内，也可能只是 padding（填充）或 garbage value（垃圾值）。

`valid_d` 防止超过实际 `head_dim`。当前 `head_dim=block_d=128`，所以它全部为 `True`，
但保留后可完整表达边界语义。

K/V 的无效位置可以在 load 时填 `0.0`，但无效 score 还必须变成 `-inf`：

```python
scores = tl.where(valid_t, scores, -float("inf"))
```

否则：

```text
invalid K = 0
dot(0, q) = 0
exp(0) = 1
```

无效 token 虽然不会通过零 V 增加分子，却会增加 softmax 分母，使输出被错误缩小。

## 5. 当前 Tile 的 Score

$$
\mathrm{score}_t
= \left(\sum_d k_{t,d}q_d\right)\times\mathrm{scale}
$$

其中 attention scale（注意力缩放）通常为：

$$
\mathrm{scale}=\frac{1}{\sqrt{D}}
$$

```python
scores = tl.sum(
    k_tile * q_vec[None, :],
    axis=1,
) * scale
```

shape trace：

```text
k_tile:                 [T, D]
q_vec[None, :]:         [1, D]
相乘:                    [T, D]
sum(axis=1), 消去 D:    [T]
```

这等价于：

```text
k_tile [T, D] @ q_vec [D] -> scores [T]
```

## 6. Tile 局部摘要

$$
m_{\mathrm{tile}}=\max_t \mathrm{score}_t
$$

$$
p_{\mathrm{tile},t}=\exp(\mathrm{score}_t-m_{\mathrm{tile}})
$$

$$
l_{\mathrm{tile}}=\sum_t p_{\mathrm{tile},t}
$$

$$
\mathrm{acc}_{\mathrm{tile}}=\sum_t p_{\mathrm{tile},t}v_t
$$

```text
m_tile:   scalar
p_tile:   [T]
l_tile:   scalar
acc_tile: [D]
```

`acc_tile` 的 shape：

```text
p_tile[:, None]:             [T, 1]
v_tile:                      [T, D]
相乘:                         [T, D]
sum(axis=0), 消去 T:         [D]
```

完全无效的尾 tile 会出现 `-inf - (-inf)`。因此代码显式使用：

```python
p_tile = tl.where(
    valid_t,
    tl.exp(scores - m_tile),
    0.0,
)
```

## 7. 合并 Running State

旧状态基于 `running_max`，当前 tile 基于 `m_tile`。两组指数参考最大值不同，必须先转换到
共同的 `m_new`：

$$
m_{\mathrm{new}}=\max(m_{\mathrm{old}},m_{\mathrm{tile}})
$$

$$
\mathrm{old\_scale}=\exp(m_{\mathrm{old}}-m_{\mathrm{new}})
$$

$$
\mathrm{tile\_scale}=\exp(m_{\mathrm{tile}}-m_{\mathrm{new}})
$$

$$
l_{\mathrm{new}}
=l_{\mathrm{old}}\,\mathrm{old\_scale}
+l_{\mathrm{tile}}\,\mathrm{tile\_scale}
$$

$$
\mathrm{acc}_{\mathrm{new}}
=\mathrm{acc}_{\mathrm{old}}\,\mathrm{old\_scale}
+\mathrm{acc}_{\mathrm{tile}}\,\mathrm{tile\_scale}
$$

代码保留两路贡献的换行：

```python
l_new = (
    running_sum * old_scale
    + l_tile * tile_scale
)

acc_new = (
    accumulator * old_scale
    + acc_tile * tile_scale
)
```

这里的 `old_scale/tile_scale` 与 attention 的 `1/sqrt(D)` 不同：

```text
attention scale:          缩放 q·k，控制 score 量级
old_scale / tile_scale:   把两个摘要转换到相同指数参考最大值
```

## 8. `tl.max` 与 `tl.maximum`

`tl.max` 是 reduction（归约）：

```python
m_tile = tl.max(scores, axis=0)  # [T] -> scalar
```

`tl.maximum` 是 element-wise maximum（逐元素最大值）：

```python
m_new = tl.maximum(running_max, m_tile)
```

例如：

```text
maximum([2, 8, 3], [5, 1, 3]) = [5, 8, 3]
```

可以记为：

```text
tl.max(x, axis=...): 从 x 内部沿 axis 找最大值，发生归约
tl.maximum(a, b):    a 和 b 对应位置比较，支持广播
```

## 9. 最终输出

$$
\mathrm{out}=\frac{\mathrm{accumulator}}{\mathrm{running\_sum}}
$$

`running_max` 不直接出现在最终输出中，因为分子和分母使用了相同的指数缩放，做除法时会
抵消。它只负责 numerical stability（数值稳定性）。

输出地址与 Q 相同：

```python
out_ptrs = (
    out_ptr
    + (batch_idx * num_heads + head_idx) * head_dim
    + offs_d
)

tl.store(out_ptrs, output, mask=valid_d)
```

## 10. 完整数据流

```text
固定 (batch_idx, head_idx)
        |
加载 q[b,h,:]                              [D]
        |
顺序遍历 context tile
        |
加载 k/v[b,tile,h,:]                       [T,D]
        |
计算 scores                                [T]
        |
生成 m_tile/l_tile/acc_tile                scalar/scalar/[D]
        |
合并 running_max/running_sum/accumulator
        |
全部 tile 完成
        |
output = accumulator / running_sum         [D]
        |
写入 out[b,h,:]
```

一句话复述：

```text
每个 Triton program 固定一个 (batch, head)，分 tile 读取整个 context 的 K/V，
用 online softmax 维护 max、exp sum 和 weighted V accumulator，最后归一化并写出
out[batch, head, :]。
```

## 11. 避免的中间张量

普通分步实现可能物化完整：

```text
scores:              [S]
probs:               [S]
probs[:, None] * V:  [S, D]
```

当前 kernel 每次只保留 tile 数据和 running state：

```text
scores / p_tile:        [T]
k_tile / v_tile:        [T, D]
running_max/sum:        scalar
accumulator:            [D]
```

关键收益是融合 K/V 读取、score、softmax 和 V 加权，避免完整 scores/probs 的显存写回与
再次读取。它仍然需要读取有效 K/V。

## 12. 验证与剩余限制

GPU correctness tests（GPU 正确性测试）覆盖：

```text
单 tile 和跨 tile
最后 tile 未填满
完全无效的尾 tile
multi-batch / multi-head
context_len 后填 garbage values
```

```text
GPU tests:     3 passed
default tests: 14 passed, 3 GPU tests deselected
GPU:           NVIDIA GeForce RTX 5090
```

仍未覆盖：

```text
paged KV / block table
head_dim=64
BF16
context_len=0
non-contiguous tensors
benchmark / profiling
```
