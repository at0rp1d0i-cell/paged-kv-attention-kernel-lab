# Online Softmax 学习笔记

这份笔记记录 Week 2 进入 Triton kernel（Triton 内核）前需要掌握的
online softmax（在线 softmax）推导。它对应当前代码里的
`dense_decode_attention_online`。

写法原则：

- 数学定义和推导使用公式块。
- 代码实现、shape trace（形状追踪）和程序结构使用代码块。
- 公式讲语义，代码块讲如何落到 PyTorch / Triton 风格实现。

## 1. 为什么需要 online softmax

decode attention（解码注意力）里 `q_len = 1`。对一个 batch item 和一个
head 来说，dense attention（稠密注意力）的核心计算是：

$$
\mathrm{score}_t = q \cdot k_t \times \mathrm{scale}
$$

$$
\mathrm{prob}_t =
\frac{\exp(\mathrm{score}_t)}
{\sum_j \exp(\mathrm{score}_j)}
$$

$$
\mathrm{out} =
\sum_t \mathrm{prob}_t \, v_t
$$

其中 `t` 是 token position（token 位置）下标。

如果一次性处理全部 context（上下文），可以直接 materialize（物化）出完整分数向量：

```text
scores: [T]
```

然后调用 softmax。但长 context 下，Triton kernel 不应该依赖完整 `scores`
临时数组，而是按 tile（分块）流式扫描 K/V：

```text
tile 0: token 0..block_size-1
tile 1: token block_size..2*block_size-1
...
```

online softmax 的目标是：不保存完整 `scores`，也能得到和 dense softmax
数学等价的结果。

## 2. 稳定 softmax 的三个状态量

为了避免 `exp(score)` overflow（上溢），稳定 softmax 会减去全局最大值：

$$
m = \max_i \mathrm{score}_i
$$

$$
\mathrm{softmax}(\mathrm{score}_i) =
\frac{\exp(\mathrm{score}_i - m)}
{\sum_j \exp(\mathrm{score}_j - m)}
$$

attention output 可以写成：

$$
l =
\sum_i \exp(\mathrm{score}_i - m)
$$

$$
\mathrm{acc} =
\sum_i \exp(\mathrm{score}_i - m) \, v_i
$$

$$
\mathrm{out} =
\frac{\mathrm{acc}}{l}
$$

所以 online softmax 只需要维护三个 running state（运行状态）：

```text
m:   [H]      running max
l:   [H]      running exp sum
acc: [H, D]   running weighted value sum
```

其中：

```text
H = num_heads（注意力头数量）
D = head_dim（每个 head 的维度）
```

## 3. 当前 tile 的局部摘要

在当前实现中，一个 tile 的输入 shape（形状）是：

```text
q_b:    [H, D]
k_tile: [TILE, H, D]
v_tile: [TILE, H, D]
```

当前 tile 的 score 计算和 dense 版本完全一样，只是 token 范围从全部
`valid_len` 变成当前 tile：

```python
scores_tile = (k_tile * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale
```

shape trace（形状追踪）：

```text
k_tile:             [TILE, H, D]
q_b.unsqueeze(0):   [1, H, D]
相乘后:              [TILE, H, D]
sum(dim=-1):        [TILE, H]
transpose(0, 1):    [H, TILE]
scores_tile:        [H, TILE]
```

然后把当前 tile 压缩成一个局部摘要。

每个 head 在当前 tile 内的最大值：

$$
m_{\mathrm{tile}}[h] =
\max_t \mathrm{scores}_{\mathrm{tile}}[h, t]
$$

当前 tile 在 `m_tile` 尺度下的未归一化权重：

$$
p_{\mathrm{tile}}[h, t] =
\exp\left(
\mathrm{scores}_{\mathrm{tile}}[h, t]
- m_{\mathrm{tile}}[h]
\right)
$$

当前 tile 的局部分母：

$$
l_{\mathrm{tile}}[h] =
\sum_t p_{\mathrm{tile}}[h, t]
$$

当前 tile 的局部加权 V 分子：

$$
\mathrm{acc}_{\mathrm{tile}}[h, :] =
\sum_t p_{\mathrm{tile}}[h, t] \,
v_{\mathrm{tile}}[t, h, :]
$$

对应代码 shape 是：

```text
m_tile:   [H]
p_tile:   [H, TILE]
l_tile:   [H]
acc_tile: [H, D]
```

这里的 `p_tile` 不是最终 probability（概率），而是当前 tile 在
`m_tile` 尺度下的 numerator weight（分子权重）。

代码里 `acc_tile` 需要先把 V 转成按 head 对齐：

```python
v_tile_by_head = v_tile.transpose(0, 1)  # [H, TILE, D]
acc_tile = (p_tile.unsqueeze(-1) * v_tile_by_head).sum(dim=1)
```

## 4. 为什么要用 m_new 重缩放

旧状态是基于 `m_old` 算的：

$$
l_{\mathrm{old}} =
\sum_{i \in \mathrm{old}}
\exp(\mathrm{score}_i - m_{\mathrm{old}})
$$

$$
\mathrm{acc}_{\mathrm{old}} =
\sum_{i \in \mathrm{old}}
\exp(\mathrm{score}_i - m_{\mathrm{old}}) \, v_i
$$

当前 tile 是基于 `m_tile` 算的：

$$
l_{\mathrm{tile}} =
\sum_{i \in \mathrm{tile}}
\exp(\mathrm{score}_i - m_{\mathrm{tile}})
$$

$$
\mathrm{acc}_{\mathrm{tile}} =
\sum_{i \in \mathrm{tile}}
\exp(\mathrm{score}_i - m_{\mathrm{tile}}) \, v_i
$$

但合并后，旧 token 和当前 tile 必须统一到同一个最大值尺度：

$$
m_{\mathrm{new}} =
\max(m_{\mathrm{old}}, m_{\mathrm{tile}})
$$

旧部分要从 `m_old` 尺度换到 `m_new` 尺度：

$$
\exp(\mathrm{score}_i - m_{\mathrm{new}})
=
\exp(\mathrm{score}_i - m_{\mathrm{old}})
\times
\exp(m_{\mathrm{old}} - m_{\mathrm{new}})
$$

当前 tile 也要从 `m_tile` 尺度换到 `m_new` 尺度：

$$
\exp(\mathrm{score}_i - m_{\mathrm{new}})
=
\exp(\mathrm{score}_i - m_{\mathrm{tile}})
\times
\exp(m_{\mathrm{tile}} - m_{\mathrm{new}})
$$

因此更新式是：

$$
\mathrm{old\_scale} =
\exp(m_{\mathrm{old}} - m_{\mathrm{new}})
$$

$$
\mathrm{tile\_scale} =
\exp(m_{\mathrm{tile}} - m_{\mathrm{new}})
$$

$$
l_{\mathrm{new}} =
l_{\mathrm{old}} \times \mathrm{old\_scale}
+
l_{\mathrm{tile}} \times \mathrm{tile\_scale}
$$

$$
\mathrm{acc}_{\mathrm{new}} =
\mathrm{acc}_{\mathrm{old}} \times \mathrm{old\_scale}[:, None]
+
\mathrm{acc}_{\mathrm{tile}} \times \mathrm{tile\_scale}[:, None]
$$

最后：

$$
\mathrm{out} =
\frac{\mathrm{acc}}{l[:, None]}
$$

## 5. 关于“能不能直接根据 tile 和 old 更新”

可以，但要分清楚“直接更新”的对象。

对 `m` 来说，确实可以直接写：

```python
m_new = torch.maximum(m, scores_tile.max(dim=-1).values)
```

这和先写：

```python
m_tile = scores_tile.max(dim=-1).values
m_new = torch.maximum(m, m_tile)
```

完全等价。单独拆出 `m_tile` 只是为了让局部 tile summary（tile 摘要）
更清楚。

但对 `l` 和 `acc` 来说，不能只把当前 tile 的贡献简单加到旧状态上：

```text
l = l_old + l_tile        # 错
acc = acc_old + acc_tile  # 错
```

原因是：

```text
l_old / acc_old   使用的是 m_old 尺度
l_tile / acc_tile 使用的是 m_tile 尺度
```

两个尺度不同，必须先统一到 `m_new` 尺度后才能相加。

所以有两种等价写法。

写法 A：先形成 tile 摘要，再缩放合并：

```python
m_tile = scores_tile.max(dim=-1).values
p_tile = torch.exp(scores_tile - m_tile.unsqueeze(-1))
l_tile = p_tile.sum(dim=-1)
acc_tile = (p_tile.unsqueeze(-1) * v_tile_by_head).sum(dim=1)

m_new = torch.maximum(m, m_tile)
old_scale = torch.exp(m - m_new)
tile_scale = torch.exp(m_tile - m_new)

l = l * old_scale + l_tile * tile_scale
acc = acc * old_scale.unsqueeze(-1) + acc_tile * tile_scale.unsqueeze(-1)
m = m_new
```

写法 B：先算 `m_new`，当前 tile 直接用 `m_new` 尺度：

```python
m_tile = scores_tile.max(dim=-1).values
m_new = torch.maximum(m, m_tile)

old_scale = torch.exp(m - m_new)
p_tile_new_scale = torch.exp(scores_tile - m_new.unsqueeze(-1))

l = l * old_scale + p_tile_new_scale.sum(dim=-1)
acc = (
    acc * old_scale.unsqueeze(-1)
    + (p_tile_new_scale.unsqueeze(-1) * v_tile_by_head).sum(dim=1)
)
m = m_new
```

这两种写法数学等价。当前代码采用写法 A，因为它更明确地表达：

```text
每个 tile 可以先独立压缩成 (m_tile, l_tile, acc_tile)，
再和历史 running state 合并。
```

这对后续理解 Triton kernel 的循环结构更有帮助。

## 6. 树形合并和后续 split-KV

把每个 tile 表示成 `(m_tile, l_tile, acc_tile)` summary（摘要）还有一个更重要的意义：
它让 attention 的 context 维度具备 tree reduction（树形归约）的形式。

如果每个 tile 或每个 context segment（上下文分段）都先独立算出：

```text
m_part:   [H]
l_part:   [H]
acc_part: [H, D]
```

那么任意两个 partial summary（部分摘要）都可以用同一套公式合并。

合并后的最大值：

$$
m_{\mathrm{new}} =
\max(m_a, m_b)
$$

合并后的分母：

$$
l_{\mathrm{new}} =
l_a \times \exp(m_a - m_{\mathrm{new}})
+
l_b \times \exp(m_b - m_{\mathrm{new}})
$$

合并后的加权 V 分子：

$$
\mathrm{acc}_{\mathrm{new}} =
\mathrm{acc}_a \times \exp(m_a - m_{\mathrm{new}})
+
\mathrm{acc}_b \times \exp(m_b - m_{\mathrm{new}})
$$

这意味着合并不一定只能顺序执行：

```text
tile0   tile1   tile2   tile3
  |       |       |       |
summary summary summary summary
   \     /         \     /
  merge01         merge23
        \         /
        merge_all
```

当前 Week 2 的 `dense_decode_attention_online` 和后续 Triton v0 仍然是 sequential scan
（顺序扫描）：一个 `(batch, head)` 逻辑上按 token/tile 顺序扫完整个 context。

但这个 summary 形式会服务后续的 split-KV（分段 KV）优化：

```text
program_id = (batch, head, context_segment)
```

也就是把长 context 拆成多个 segment，让不同 program 并行计算各自的
`(m_part, l_part, acc_part)`，再用 reduce kernel（归约内核）或 reduce stage（归约阶段）
合并 partial summaries。

这种方法主要解决：

```text
batch 小 + context 很长
```

时的 SM occupancy（SM 占用率）不足问题。原来只有：

```text
B * H
```

个 program；split-KV 后变成：

```text
B * H * num_segments
```

个 program，从而增加并行度。

需要注意的是，tile/segment summary 不能在 query `q` 出现前预计算，因为：

$$
\mathrm{scores} = q \cdot k
$$

`m/l/acc` 都依赖当前 decode step（解码步）的 `q`。能并行的是：同一个 `q`
已经确定后，不同 context segment 同时计算自己的 summary。

这个优化也有代价：

```text
1. 需要中间 buffer 保存 partial m/l/acc。
2. 需要额外 reduce kernel 或 reduce stage。
3. acc_part 的 shape 是 [H, D]，head_dim=128 时中间写回成本不小。
4. 对短 context 或大 batch，原本并行度已经足够，split-KV 可能不划算。
```

所以当前阶段先掌握 `(m, l, acc)` 的可合并性；真正是否采用 split-KV，要等 Week 4
benchmark（基准测试）和 profiling（性能剖析）证明瓶颈后再决定。

## 7. 初始化为什么成立

循环开始前：

$$
m = -\infty,\quad l = 0,\quad \mathrm{acc} = 0
$$

第一块进来时：

$$
m_{\mathrm{new}} =
\max(-\infty, m_{\mathrm{tile}})
= m_{\mathrm{tile}}
$$

$$
\mathrm{old\_scale} =
\exp(-\infty - m_{\mathrm{new}})
= 0
$$

$$
\mathrm{tile\_scale} =
\exp(m_{\mathrm{tile}} - m_{\mathrm{new}})
= 1
$$

所以：

$$
l_{\mathrm{new}} =
0 + l_{\mathrm{tile}}
$$

$$
\mathrm{acc}_{\mathrm{new}} =
0 + \mathrm{acc}_{\mathrm{tile}}
$$

这说明第一块自然成为当前 running state。

## 8. 当前代码位置

当前实现位于：

```text
src/paged_kv_attention/reference.py::dense_decode_attention_online
```

测试位于：

```text
tests/test_reference.py::test_dense_decode_attention_online_matches_dense_across_tiles
```

这个测试验证了：

```text
dense_decode_attention_online == dense_decode_attention
```

覆盖 FP16 输入、multi-batch（多 batch）、multi-head（多头）、跨多个 tile 的情况。
