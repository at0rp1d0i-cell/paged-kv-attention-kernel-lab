# Triton Paged KV Indexing 学习笔记

这份笔记只记录 continuous dense KV（连续稠密 KV）到 paged KV（分页 KV）的增量变化。
attention 数学、online softmax（在线 softmax）、program mapping（程序映射）和输出布局均
保持不变。

## 1. Dense 与 Paged 的核心差异

Dense K/V 直接使用逻辑 token `t`：

```text
k[b, t, h, d]
```

Paged K/V 需要先把逻辑 token 映射到物理存储位置：

```text
t
|
+-- logical_block = t // block_size
+-- slot = t % block_size
|
+-- physical_block = block_tables[b, logical_block]
|
+-- k_cache[physical_block, slot, h, d]
```

因此 paged kernel 真正新增的是 indirect lookup（间接查表）和随后的地址计算。

## 2. Block Table 查表

`block_tables` 的 layout（布局）是：

```text
[B, max_blocks_per_seq]
```

当前 tile 中：

```python
logical_blocks = offs_t // block_size  # [T]
slots = offs_t % block_size            # [T]

block_table_ptrs = (
    block_tables_ptr
    + batch_idx * max_blocks_per_seq
    + logical_blocks
)  # [T]

physical_blocks = tl.load(
    block_table_ptrs,
    mask=valid_t,
    other=0,
)  # [T]
```

变量语义：

```text
block_table_ptrs: 地址 [T]
physical_blocks:  block table 中加载出的整数数据 [T]
```

无效 token 必须 mask 掉 block table load，否则尾 tile 可能使用越界的
`logical_blocks` 读取不存在的表项。

## 3. Paged K/V Offset

K/V cache layout 是：

```text
[num_blocks, block_size, H, D]
```

元素 `k_cache[physical_block, slot, h, d]` 的线性偏移是：

$$
\operatorname{offset}
=(((\mathrm{physical\_block}\times\mathrm{block\_size}+\mathrm{slot})H+h)D+d)
$$

Triton 代码：

```python
offs_kv = (
    ((physical_blocks[:, None] * block_size + slots[:, None])
    * num_heads + head_idx)
    * head_dim + offs_d[None, :]
)  # [T, D]
```

shape trace：

```text
physical_blocks[:, None]: [T, 1]
slots[:, None]:           [T, 1]
offs_d[None, :]:          [1, D]
offs_kv:                  [T, D]
```

## 4. 哪些部分不变

从 dense kernel 到 paged kernel，以下内容不变：

```text
grid = (B, H)
Q 的加载
context tile 循环
score 计算
online softmax 的 m/l/acc 更新
out[b,h,:] 的连续输出地址
```

只改变：

```text
K/V 输入 layout
增加 block_tables
logical token 到 physical block/slot 的地址链
对应的 wrapper contract 和 correctness tests
```

特别注意：paged 只影响 K/V 的读取，`out` 仍是 `[B,H,D]` 的连续 tensor，不能使用
`physical_blocks/slots` 计算输出地址。

## 5. Garbage Block/Slot 测试

测试将未使用 physical block 和最后 block 的未使用 slot 填成很大的 garbage values
（垃圾值）。这样以下 bug 会明显污染输出：

```text
block table 查错 physical block
slot 计算错误
读取 context_len 之后的位置
最后 tile 的 mask 错误
```

如果无效位置只填零，即使 kernel 错误读取，结果也可能碰巧接近正确值，从而漏报 bug。

## 6. 当前验证

Paged GPU tests 覆盖：

```text
随机乱序 physical block
跨多个 logical block
最后 block 未填满
完全无效的尾 tile
multi-batch / multi-head
未使用 block 和 slot 填垃圾值
```

当前结果：

```text
tests/test_triton_decode.py: 6 passed（dense 3 + paged 3）
default CPU tests:          14 passed, 6 GPU tests deselected
```

一句话复述：

```text
Paged Triton decode 不改变 attention 数学，只把 logical token t 转换为
(physical_block, slot)，再复用连续 kernel 的 online softmax 主体。
```
