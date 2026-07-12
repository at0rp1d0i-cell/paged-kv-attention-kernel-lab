# Triton 与 PyTorch 的区别

Triton 和 PyTorch 都允许用接近 Python 的表达描述 tensor（张量）计算，但它们位于不同抽象
层级。PyTorch 主要表达“对整个 tensor 做什么”，Triton 主要表达“一个 GPU program 如何
处理输出的一块数据”。

## 1. 核心对比

| 维度 | PyTorch | Triton |
| --- | --- | --- |
| 主要抽象 | tensor operation（张量操作） | program instance（程序实例）与 tile |
| 并行划分 | framework/operator 内部决定 | 作者通过 grid 和 `program_id` 决定 |
| 内存访问 | 高级索引和已有算子 | pointer、offset、mask、`tl.load/store` |
| 中间结果 | 容易物化为新 tensor | 可在单个 kernel 内保留为寄存器/局部值 |
| 融合 | 依赖已有 fused op 或 compiler | 可以显式把多个步骤写进同一 kernel |
| 边界处理 | 许多算子自动处理 | 常需显式 mask |
| 正确性调试 | eager execution（即时执行）方便检查 | JIT 后在 GPU 上验证，错误更接近底层 |
| 性能控制 | 主要选择算子、layout、dtype | 还要选择 mapping、tile、warps 和访存方式 |

## 2. 同一个计算的两种视角

PyTorch 通常从完整 tensor 出发：

```python
scores = (k * q.unsqueeze(0)).sum(dim=-1) * scale
probs = torch.softmax(scores, dim=0)
out = (probs.unsqueeze(-1) * v).sum(dim=0)
```

这段代码描述完整数据流，但每一步可能对应一个或多个 kernel launch（内核启动），并可能
产生完整 `scores/probs` 中间 tensor。

Triton 先决定一个 program 负责哪块输出：

```text
program_id = (batch, head)
owned output = out[batch, head, :]
```

然后 program 自己：

```text
计算地址
按 tile 加载 K/V
计算 score
更新 online softmax
写回最终 output
```

这样可以把原本跨多个算子的工作融合进一个 kernel。

## 3. Triton 中需要显式思考的内容

PyTorch 中常被隐藏的细节，在 Triton 中成为实现的一部分：

```text
1. grid 中需要多少 program？
2. 每个 program 拥有哪块输出？
3. tensor layout 对应什么线性 offset？
4. 哪些 load/store 可能越界，需要什么 mask？
5. 哪个维度做 reduction？中间值是什么 shape？
6. 哪些中间结果能留在 kernel 内，避免显存往返？
```

本项目中的例子：

```text
PyTorch: k[b, t, h, :]

Triton:
offs_kv -> k_ptrs -> tl.load -> k_tile
```

## 4. 为什么目前感觉 Triton 不难

当前 kernel 之所以比较容易进入，是因为：

```text
q_len = 1
head_dim 固定为 128
一个 program 对应一个 (batch, head)
online softmax 已经先在 PyTorch 中推导和验证
dense 到 paged 只改变 K/V 地址计算
```

也就是说，我们先把数学、shape 和接口拆开验证，再翻译成 Triton。此时 Triton 语法只是
对已有推导的低层表达，并没有同时承担算法设计。

## 5. 真正困难通常在哪里

能写出正确 Triton kernel 不等于已经写出高性能 kernel。后续难点通常包括：

```text
program mapping 是否提供足够并行度
K/V load 是否 coalesced（合并访存）
tile size 对寄存器压力和 occupancy 的影响
num_warps / num_stages 的选择
长 context、小 batch 下是否需要 split-KV
paged indirect lookup 的额外成本
不同 GPU、dtype、head_dim 下的性能稳定性
如何用 benchmark/profiling 证明瓶颈和优化收益
```

因此可以把学习分为三层：

```text
第一层：写对数学和 shape
第二层：写对 mapping、pointer 和 mask
第三层：用 benchmark/profiling 把 kernel 写快并解释原因
```

当前已经完成前两层的第一轮，下一阶段开始进入第三层及更通用的 correctness coverage。

## 6. 选择 PyTorch 还是 Triton

优先使用 PyTorch 的场景：

```text
已有高质量算子能直接表达需求
计算不是性能瓶颈
需要快速迭代、动态控制流和广泛 dtype/layout 支持
```

考虑 Triton 的场景：

```text
多个简单操作之间存在明显中间显存往返
需要自定义数据布局或间接寻址
现有算子无法有效融合
目标 workload 形状稳定，值得专门优化
可以投入 correctness、benchmark 和 profiling 成本
```

一句话总结：

```text
PyTorch 让人从完整 tensor 和算子组合思考；Triton 让人从 GPU program、tile、地址和
数据移动思考。Triton 的语法不一定难，难的是为具体 workload 做出正确且可证明的性能决策。
```
