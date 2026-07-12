# Week 2 Lab Notes

## 本周目标

- 用 PyTorch block-wise online softmax（分块在线 softmax）连接 dense reference 与 Triton。
- 在连续 `[B,S,H,D]` KV layout 上完成第一个 Triton decode attention kernel。
- 先锁定 `head_dim=128`、FP16 input、FP32 accumulation，避免同时调试 paged indexing。

## 最难的理解点

`p_tile` 只是以 `m_tile` 为参考最大值计算的分子权重，不是最终 probability（概率）。旧状态
和当前 tile 分别基于 `running_max`、`m_tile`，必须先转换到 `m_new` 的共同尺度后才能合并。

Triton pointer arithmetic（指针运算）也需要同时理解逻辑索引、线性布局和广播：
`offs_t[:,None]` 与 `offs_d[None,:]` 构造 `[T,D]` 偏移，和基础指针相加得到地址，再由
`tl.load` 取得数据。

## 亲手 Debug 的问题

- 混淆 `tl.max` 和 `tl.maximum`：前者沿 axis 做 reduction（归约），后者逐元素比较两个
  输入。`m_tile` 使用 `tl.max`，`m_new` 使用 `tl.maximum`。
- 一度在定义 `m_new` 之前计算 `old_scale/tile_scale`，Ruff 的 undefined-name 检查暴露了
  状态更新顺序错误。
- K/V 的无效 load 使用 `other=0.0` 还不够。无效 K 会产生 score `0`，并通过 `exp(0)`
  错误增加 softmax 分母，因此还要把无效 score 改成 `-inf`。
- 完全无效的尾 tile 会出现 `-inf - (-inf)`，代码通过 `tl.where` 明确把无效 `p_tile`
  设为零，GPU 测试覆盖了这个场景。
- 自动 formatter 会压缩 online softmax 两路合并公式。核心公式使用局部
  `# fmt: off/on` 保留更适合学习复盘的换行。

## 验证结果

GPU correctness tests（GPU 正确性测试）在 RTX 5090 上与 FP32 dense reference 对齐，
覆盖单 tile、跨 tile、partial/full invalid tail tile、multi-batch、multi-head，以及
`context_len` 后的 garbage values。

```text
pytest -q tests/test_triton_decode.py: 6 passed
bash scripts/run_tests.sh:             14 passed, 6 deselected
```

## 当前理解

每个 Triton program 固定一个 `(batch, head)`，向量化处理完整 `D`，并顺序扫描 context
tile。kernel 内只维护 scalar `running_max/running_sum` 和 `[D]` accumulator，最后计算
`accumulator / running_sum` 并写入 `out[b,h,:]`。

当前 kernel 避免物化完整 `[S]` scores/probs，并把 K/V 读取、score、softmax 和 V 加权
融合到一次 kernel launch 中。它仍然需要读取全部有效 K/V；小 batch、长 context 时，
program 数量只有 `B*H`，可能出现并行度不足。

## Git Checkpoint

当前节点定义为：

```text
continuous dense Triton decode correctness
```

节点成立的证据：核心 kernel 可 launch、边界 mask 通过 garbage-value 测试、输出与 FP32
dense reference 对齐。Git 整理按实现、测试、笔记的语义拆分，不使用 `v0 done` 等临时名称。

第二个节点定义为：

```text
paged Triton decode correctness
```

这个节点只改变 K/V 的寻址：logical token 先拆成 logical block 和 slot，再通过 block table
映射到 physical block。attention 数学、online softmax、program mapping 和输出布局均复用
连续版本。随机 block table 与 garbage block/slot 测试证明 kernel 没有误读无效物理位置。

## 方向感受

Triton 的基础语法比预想中直接：已有 PyTorch 数学推导和 shape trace 后，`tl.load`、
reduction、mask 和 online softmax 可以逐步翻译。当前感受到的主要难度不是语法，而是
pointer arithmetic 和边界语义。

需要保留判断：正确 kernel 只是第一步。后续 program mapping、coalesced memory access
（合并访存）、寄存器压力、occupancy、tile 参数与 profiling 才是性能工程的核心难点。

## 下一步与剩余风险

- 后续补 `head_dim=64`、BF16、`context_len=0` 和更完整的 tolerance coverage。
- 先做一个最小 benchmark，确认 dense/paged kernel 的延迟量级和 context scaling。
- 分析 block size、block_t、batch 和 context length 对并行度与访存的影响。
- split-KV 是否值得实现，要由长 context benchmark（基准测试）和 profiling（性能剖析）
  证明。
