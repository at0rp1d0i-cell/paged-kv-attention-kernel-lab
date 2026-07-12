# Triton 常用语法速查

这份笔记只收录 paged KV attention 项目中已经使用或近期会使用的 Triton 语法。Triton
代码虽然写在 Python 文件中，但 `@triton.jit` 函数会被 JIT compilation（即时编译）为
GPU kernel，不是普通 Python 函数。

## 1. Kernel 定义

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    ...
```

普通参数可以来自 runtime（运行时）；`tl.constexpr` 参数在编译期已知，可用于静态 shape、
循环范围和 specialization（特化）。

本项目中的典型编译期参数：

```python
head_dim: tl.constexpr
block_t: tl.constexpr
block_d: tl.constexpr
block_size: tl.constexpr
```

## 2. Grid 与 Kernel Launch

```python
grid = (batch_size, num_heads)

_kernel[grid](
    q,
    k,
    v,
    out,
    num_heads=num_heads,
    head_dim=head_dim,
    block_t=block_t,
)
```

`grid` 决定创建多少个 program instance（程序实例）。它不是 tensor shape，也不是 CUDA
thread 的直接数量。

一维 grid：

```python
grid = (triton.cdiv(n_elements, block_size),)
```

`triton.cdiv(a, b)` 表示向上取整除法。

## 3. Program ID

```python
pid = tl.program_id(axis=0)
batch_idx = tl.program_id(axis=0)
head_idx = tl.program_id(axis=1)
```

对于：

```python
grid = (B, H)
```

每个 program 获得一个 `(batch_idx, head_idx)`，并负责 `out[batch_idx, head_idx, :]`。

## 4. `tl.arange`

```python
offs_d = tl.arange(0, block_d)  # [D]
offs_t = start + tl.arange(0, block_t)  # [T]
```

`tl.arange` 生成一组编译期可知长度的连续逻辑索引。它生成的是 offset/index（偏移/索引），
不是从显存读取的数据。

Triton 对 block shape 有编译约束，常用 block size 通常选择 2 的幂。

## 5. Triton 中的 Broadcasting

和 PyTorch 类似，可增加大小为 1 的维度：

```python
offs_t[:, None]  # [T] -> [T, 1]
offs_d[None, :]  # [D] -> [1, D]
```

```text
[T,1] + [1,D] -> [T,D]
```

例如构造 dense K/V 地址：

```python
offs_kv = (
    ((batch_idx * max_context_len + offs_t[:, None])
    * num_heads + head_idx)
    * head_dim + offs_d[None, :]
)  # [T, D]
```

## 6. Pointer、Offset 与 Data

```text
k_ptr:    整个 K tensor 的基础指针
offs_kv:  当前 tile 的线性偏移 [T,D]
k_ptrs:   当前 tile 的地址集合 [T,D]
k_tile:   地址中加载出的数据 [T,D]
```

```python
k_ptrs = k_ptr + offs_kv
k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
```

Triton 中的 pointer arithmetic（指针运算）以元素为单位；PyTorch tensor 参数会作为基础
指针传入 kernel。

## 7. `tl.load`

```python
x = tl.load(
    x_ptrs,
    mask=valid,
    other=0.0,
)
```

含义：

```text
mask=True:  从对应地址加载
mask=False: 不读取该地址，返回 other
```

`other` 应与后续语义匹配：

```text
K/V 无效 load: other=0.0
block table 无效 load: other=0
```

mask 不只是让结果为零，更重要的是阻止实际的越界读取。

## 8. `tl.store`

```python
tl.store(
    out_ptrs,
    output,
    mask=valid_d,
)
```

地址、数据和 mask 的 shape 必须可兼容。paged K/V 只改变输入地址，输出仍是连续 `[B,H,D]`
布局：

```python
out_ptrs = (
    out_ptr
    + (batch_idx * num_heads + head_idx) * head_dim
    + offs_d
)  # [D]
```

## 9. Mask 与逻辑运算

```python
valid_t = offs_t < context_len  # [T]
valid_d = offs_d < head_dim     # [D]

kv_mask = valid_t[:, None] & valid_d[None, :]  # [T,D]
```

常用比较与逻辑运算：

```python
x < limit
x >= 0
mask_a & mask_b
mask_a | mask_b
```

不要使用 Python 的 `and/or` 组合 Triton tensor mask；使用逐元素 `&` 和 `|`。

## 10. `tl.where`

```python
scores = tl.where(valid_t, scores, -float("inf"))
p_tile = tl.where(valid_t, tl.exp(scores - m_tile), 0.0)
```

`tl.where(mask, a, b)` 逐位置选择数据。它适合改变计算语义，但不能替代 `tl.load/store`
的内存安全 mask。

例如：先越界 load，再用 `tl.where` 清零，仍然是不安全的；必须在 load 时 mask。

## 11. Reduction

```python
scores = tl.sum(k_tile * q_vec[None, :], axis=1)  # [T,D] -> [T]
m_tile = tl.max(scores, axis=0)                   # [T] -> scalar
l_tile = tl.sum(p_tile, axis=0)                   # [T] -> scalar
acc_tile = tl.sum(p_tile[:, None] * v_tile, axis=0)  # [T,D] -> [D]
```

`axis` 指被消去的维度。先写出输入 shape，再判断要消去哪一维。

## 12. `tl.max` 与 `tl.maximum`

归约最大值：

```python
m_tile = tl.max(scores, axis=0)  # [T] -> scalar
```

逐元素比较：

```python
m_new = tl.maximum(running_max, m_tile)
```

```text
tl.max(x, axis=...): 从 x 内部做 reduction
tl.maximum(a, b):    a 和 b 对应位置比较，支持广播
```

## 13. 数学函数与 Dtype

```python
tl.exp(x)
tl.zeros((block_d,), dtype=tl.float32)
x.to(tl.float32)
```

本项目使用：

```text
FP16 input
FP32 score / running state / accumulator / output
```

```python
k_tile = tl.load(...).to(tl.float32)
```

显式转 FP32 可以降低 online softmax 和加权累加的数值误差。

## 14. 编译期循环

```python
max_context_len: tl.constexpr = max_blocks_per_seq * block_size

for start in range(0, max_context_len, block_t):
    ...
```

这里循环边界和步长是编译期可知的。Triton 可以静态展开或优化循环，但不同 constexpr
参数组合可能产生不同的 compiled kernel（已编译内核）。

循环内 `context_len` 可以是 runtime scalar，通过 mask 跳过当前序列的无效 token。

## 15. Paged Indirect Lookup

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

然后构造 paged K/V offset：

```python
offs_kv = (
    ((physical_blocks[:, None] * block_size + slots[:, None])
    * num_heads + head_idx)
    * head_dim + offs_d[None, :]
)  # [T,D]
```

这是 indirect addressing（间接寻址）：先加载地址映射数据，再用该数据计算下一次 load 地址。

## 16. Online Softmax 状态

```python
running_max = -float("inf")                # scalar
running_sum = 0.0                          # scalar
accumulator = tl.zeros((block_d,), tl.float32)  # [D]
```

更新：

```python
m_new = tl.maximum(running_max, m_tile)
old_scale = tl.exp(running_max - m_new)
tile_scale = tl.exp(m_tile - m_new)

running_sum = (
    running_sum * old_scale
    + l_tile * tile_scale
)

accumulator = (
    accumulator * old_scale
    + acc_tile * tile_scale
)

running_max = m_new
```

这里的状态是 kernel 内局部值，不会在每个 tile 后写回显存。

## 17. Wrapper 与 Kernel 的职责

Python wrapper 负责：

```text
shape/dtype/device/layout 检查
计算默认 scale
分配 out
定义 grid
传入 constexpr 参数并 launch
```

Triton kernel 负责：

```text
program mapping
pointer arithmetic
masked load/store
tile 计算与 reduction
```

不要把复杂 Python tensor 检查放进 `@triton.jit` kernel；也不要假设 kernel 会自动理解
PyTorch 的多维 tensor layout。

## 18. 常见错误清单

```text
把维度大小 num_blocks 当作下标 physical_block
把 logical block 直接当作 physical block
忘记给 offs_t/offs_d 增加广播维度
把 offset、pointer 和 data 混在一起
load 时不 mask，只在计算后 tl.where
K load 填零后忘记把无效 score 设为 -inf
混淆 tl.max 与 tl.maximum
在定义 m_new 前使用它
算出 new state 后忘记写回 running state
使用 paged K/V 地址计算连续 out 地址
wrapper 中忘记创建 out 或 grid
```

## 19. 尚未使用但近期会遇到

性能调优阶段会逐渐加入：

```python
_kernel[grid](..., num_warps=4, num_stages=2)
```

还可能使用：

```text
tl.multiple_of / tl.max_contiguous 等 compiler hint（编译器提示）
triton.testing.do_bench
autotune（自动调优）
```

这些 API 应在 benchmark 和 profiling 阶段结合测量学习，不需要现在提前死记。

## 20. 阅读 Triton 代码的顺序

遇到新 kernel 时，按以下顺序读：

```text
1. wrapper 输入输出与限制
2. grid 和 program_id
3. 一个 program 拥有的输出区域
4. tl.arange 生成的逻辑索引及 shape
5. pointer offset 公式
6. load/store mask
7. reduction 维度
8. 局部状态和最终写回
9. block size、num_warps 等性能参数
```

这比从第一行开始逐句翻译更容易建立整体模型。
