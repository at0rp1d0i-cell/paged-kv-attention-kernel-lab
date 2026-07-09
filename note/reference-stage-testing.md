# Reference 阶段测试笔记

这份笔记记录 reference 阶段的测试思路。重点不是“让 pytest 变绿”，而是用测试把 dense reference（稠密参考实现）、paged reference（分页参考实现）和 block table（块表）语义钉牢。

## 1. 测试分层

当前 reference 阶段的测试可以分成四类：

```text
contract test（契约测试）:
    验证函数能运行，输出 shape / dtype 符合接口约定。

semantic test（语义测试）:
    验证数值结果真的符合 attention 公式或地址映射语义。

boundary test（边界测试）:
    验证 context_lens、最后一个 block 未填满、unused slot 等边界。

alignment test（对齐测试）:
    验证 paged reference 输出和 dense reference 输出一致。
```

`tests/test_week1_scaffold.py` 里原本更多是 scaffold / contract test。新增的 `tests/test_reference.py` 主要承担 semantic / alignment test。

## 2. Dense Reference 测试

`dense_decode_attention` 的目标是定义 FP32 ground truth（真值）：

```text
scores = q @ k^T / sqrt(D)
probs  = softmax(scores)
out    = probs @ v
```

### 2.1 手算小例子

第一条测试使用很小的 shape：

```text
B = 1
H = 1
D = 2
S = 3
```

测试目标：

```text
dense_decode_attention(q, k, v)
等于
softmax(q @ k^T) @ v
```

为什么不用全 1 / 2 这种太对称的数据：

```text
太对称的数据可能让维度写错、softmax 维度写错、加权错位等 bug 碰巧通过。
```

所以测试里使用不对称的小数据，让 scores 和 v 的每个 token / 维度都有差异。

### 2.2 context_lens mask 测试

第二条 dense 测试让：

```text
S = 3
context_lens = [1]
```

也就是只有 token 0 有效，token 1 / token 2 都是无效 token。

无效 token 的 K/V 里故意放很大的 garbage values（垃圾值）：

```text
9999, -9999
```

正确输出应该只等于有效 token 0 的 value：

```text
out = v[0]
```

这条测试证明：

```text
context_lens 是有效 token 边界；
t >= context_lens[b] 的 token 不会进入 softmax 或 value 加权。
```

### 2.3 Multi-batch / Multi-head 测试

第三条 dense 测试覆盖：

```text
B = 2
H = 2
context_lens = [2, 3]
```

测试目标：

```text
1. batch 之间不能串数据。
2. head 之间独立计算 attention。
3. 每个 batch 使用自己的 context_lens。
```

它防止实现只在 `B=1, H=1` 的最小 case 下碰巧正确。

## 3. Paged Reference 测试

`paged_decode_attention` 和 dense attention 的数学公式相同，差异只在 K/V 读取地址：

```text
logical_block  = t // block_size
slot           = t % block_size
physical_block = block_tables[b, logical_block]

k = k_cache[physical_block, slot]
v = v_cache[physical_block, slot]
```

所以 paged 测试的核心不是重新验证 attention 公式，而是验证：

```text
paged 地址映射读出的 K/V
和 dense layout 下的逻辑 K/V 等价。
```

## 4. Manual Block Table 对齐测试

第一条 paged 测试手工构造：

```text
block_size = 2
block_tables = [[2, 0]]
context_lens = [3]
```

含义：

```text
logical block 0 -> physical block 2
logical block 1 -> physical block 0
```

逻辑 token 映射：

```text
token 0 -> logical block 0, slot 0 -> physical block 2, slot 0
token 1 -> logical block 0, slot 1 -> physical block 2, slot 1
token 2 -> logical block 1, slot 0 -> physical block 0, slot 0
```

测试目标：

```text
paged_decode_attention(...) == dense_decode_attention(...)
```

它证明：

```text
1. physical block 可以乱序。
2. paged reference 按 block_tables 寻址。
3. 最后一个 block 的 unused slot 不会被读。
```

## 5. Variable-length Batch 对齐测试

第二条 paged 测试覆盖：

```text
B = 2
H = 2
context_lens = [3, 5]
block_size = 2
```

两个 batch 使用不同长度、不同 block table：

```text
batch 0: [4, 1, -1]
batch 1: [3, 0, 5]
```

测试目标：

```text
1. 不同 batch 可以有不同 context length。
2. 每个 batch 使用自己的 block table。
3. multi-head 情况下 paged 和 dense 仍然对齐。
```

## 6. Generated Block Table 对齐测试

第三条 paged 测试把手写 block table 换成：

```python
make_random_block_tables(...)
```

完整流程：

```text
1. 生成 dense q/k/v。
2. 用 make_random_block_tables 生成 random / non-contiguous block_tables。
3. 创建 k_cache / v_cache，并先填 garbage values。
4. 按 block_tables 把 dense k/v pack 进 paged cache。
5. 分别跑 dense_decode_attention 和 paged_decode_attention。
6. assert_close(paged_out, dense_out)。
```

这条测试证明 reference 阶段的三个组件已经串起来：

```text
make_random_block_tables
    -> paged k_cache / v_cache
    -> paged_decode_attention
    -> dense reference 对齐
```

## 7. Block Table Generator 测试

`make_random_block_tables` 本身不是 attention 测试，而是 test helper / data generator。

它需要保证：

```text
1. block_tables shape 正确。
2. unused entries 是 -1。
3. valid physical block id 在 [0, num_physical_blocks)。
4. valid physical block id 不重复。
5. 同一个 seed 可复现。
```

这个 helper 的价值是：让后续测试不只依赖手写简单 block table，而是能自动制造更接近真实 serving 场景的 random-order / non-contiguous paged layout。

## 8. 当前测试覆盖结论

当前 reference 阶段测试已经覆盖：

```text
dense attention 数学公式
context_lens mask 语义
multi-batch / multi-head dense 计算
manual block table paged-vs-dense 对齐
variable-length batch paged-vs-dense 对齐
generated block table paged-vs-dense 对齐
block table generator 基本语义
```

还没有覆盖的内容包括：

```text
GPU kernel correctness
FP16 / BF16 tolerance
benchmark（基准测试）
profiling（性能剖析）
```

这些属于后续 Triton kernel 和 benchmark 阶段。

## 9. 复述检查问题

完成本阶段后，应该能回答：

```text
1. contract test 和 semantic test 有什么区别？
2. 为什么 dense reference 要用可手算小例子？
3. 为什么 context_lens 测试要往无效 token 填垃圾值？
4. paged reference 为什么要和 dense reference 对齐？
5. block_tables 的地址公式是什么？
6. 为什么 manual block table 和 generated block table 都要测？
7. make_random_block_tables 为什么不是测试本身？
8. 当前 13 条测试分别给 reference 阶段提供了哪些保障？
```
