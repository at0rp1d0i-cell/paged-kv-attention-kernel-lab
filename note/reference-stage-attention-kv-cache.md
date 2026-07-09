# Reference 阶段前置知识：Attention、KV Cache 与 Paged KV

这份笔记记录进入 reference 阶段前需要讲清楚的概念：普通 attention（注意力）流程、prefill（预填充）和 decode（解码）的区别、KV cache（KV 缓存）为什么能用空间换时间，以及 dense reference（稠密参考实现）和 paged reference（分页参考实现）分别要验证什么。

## 1. 一次对话请求的拆解

以一次聊天会话为例，一次请求可以拆成三部分：

```text
H = 历史信息
U = 用户这一次的新输入
A = 模型这一次要生成的输出
```

在模型开始回答之前，`A` 还不存在，所以一次请求内部通常分成两段：

```text
prefill 阶段：处理 H + U
decode 阶段：一个 token 一个 token 生成 A
```

如果启用了 prompt cache（提示词缓存）或 prefix cache（前缀缓存），那么历史信息 `H` 的每一层 K/V 可能已经缓存好了。下一次请求只需要处理新增的 `U`，然后再进入 decode 阶段生成 `A`。

## 2. Attention 的基本流程

Transformer attention（Transformer 注意力）会先从输入 hidden states（隐藏状态）得到：

```text
Q = X Wq
K = X Wk
V = X Wv
```

然后计算：

```text
scores = Q @ K^T / sqrt(head_dim)
probs  = softmax(scores)
out    = probs @ V
```

直观理解：

- `Q`：当前 token 想查询什么信息。
- `K`：历史 token 提供的索引或匹配特征。
- `V`：历史 token 真正携带的内容。
- `Q @ K^T`：当前 token 和历史 token 的相关性。
- `softmax`：把相关性变成权重。
- `probs @ V`：按权重汇总历史信息。

在 causal language model（因果语言模型）里，第 `t` 个 token 只能看 `0..t`，不能看未来 token。

## 3. Head 数和 Head Dim

Multi-head attention（多头注意力）不是只算一套 attention，而是并行算多套 attention。

如果：

```text
hidden_dim = 4096
num_heads  = 32
head_dim   = 128
```

那么：

```text
hidden_dim = num_heads * head_dim
4096 = 32 * 128
```

在本项目 reference 阶段的 decode attention（解码注意力）里，常见 shape（形状）是：

```text
q: [batch, num_heads, head_dim]
k: [batch, context_len, num_heads, head_dim]
v: [batch, context_len, num_heads, head_dim]
```

单个 decode token 对长度为 `L` 的上下文做 attention，score 部分的计算量大约是：

```text
O(num_heads * L * head_dim)
```

加权 `V` 的部分也是同一数量级。为了讨论核心趋势时，常简化成：

```text
O(L)
```

## 4. Prefill 阶段做什么

prefill 阶段处理的是当前请求中已经存在的上下文：

```text
H + U
```

如果没有 prompt cache，需要一次性处理完整的 `H + U`，并在每一层把这些 token 的 K/V 写入 KV cache。

如果有 prompt cache，并且 `H` 的 K/V 已经存在，那么 prefill 主要处理新增输入 `U`：

```text
1. 为 U 计算新的 Q/K/V。
2. U 的 query attend 到 H 的 cached K/V 和 U 自己前面的 K/V。
3. 把 U 的 K/V 追加进 KV cache。
```

需要注意：prefill 的“填充”不是 padding（补齐），而是 populate cache（写入缓存）。

## 5. Decode 阶段为什么 q_len = 1

LLM 生成文本是 autoregressive（自回归）的：

```text
p(x1, x2, x3, ...) = p(x1) * p(x2 | x1) * p(x3 | x1, x2) * ...
```

也就是说，下一个 token 没生成出来之前，模型不知道它是什么，也就不能提前计算它的 Q/K/V。

所以 decode 阶段通常是：

```text
每一步只处理一个新 token。
```

每一步做：

```text
1. 计算当前 token 的 Q/K/V。
2. 用当前 token 的 Q attend 到所有已有 cached K/V。
3. 生成下一个 token。
4. 把当前 token 的 K/V 追加进 KV cache。
```

如果当前已经有：

```text
H + U + A[:i]
```

那么下一步生成时，当前 query 会读取：

```text
K_cache / V_cache = H + U + 已生成的 A 前缀
```

公式是：

```text
scores = q_current @ K_cache^T
probs  = softmax(scores)
out    = probs @ V_cache
```

这就是 one-query-to-long-context attention（单 query 对长上下文注意力）。

## 6. KV Cache 如何用空间换时间

KV cache 的核心思想类似算法题里的 memoization（记忆化缓存）或 prefix sum（前缀和）的“用空间换时间”：

```text
存下历史中间结果，避免每一步重复计算完整历史。
```

但它和 prefix sum 有一个重要区别：

```text
prefix sum 存的是压缩后的累计值；
KV cache 存的是每个历史 token 的 K/V，不能简单压缩成一个累计向量。
```

原因是每个新 token 的 query 不同，它对历史 token 的 attention 权重也不同：

```text
softmax(q_current @ K_cache^T)
```

权重取决于当前的 `q_current`，所以历史 token 的 K/V 需要逐 token 保留下来。

## 7. 有无 KV Cache 的计算量对比

假设当前上下文长度是 `L`，现在要生成下一个 token。

有 KV cache 时：

```text
当前 token 的 q attend 到前面 L 个 cached K/V
attention 计算量约 O(L)
```

更完整地写：

```text
O(num_heads * L * head_dim)
```

没有 KV cache 时，如果每次生成新 token 都重新跑完整 prefix，那么 causal attention 需要计算：

```text
1 + 2 + 3 + ... + L = L(L + 1) / 2
```

也就是：

```text
约 O(L^2 / 2)
```

所以 KV cache 把每步 decode 从“重新处理整段上下文”变成：

```text
只处理当前 token，并读取历史 K/V。
```

## 8. Dense KV Layout

Dense KV layout（稠密 KV 布局）是最直观的存储方式：

```text
k: [batch, max_context_len, num_heads, head_dim]
v: [batch, max_context_len, num_heads, head_dim]
```

也就是每条序列的 KV 在物理内存中连续存放：

```text
k[b, t, h, d]
v[b, t, h, d]
```

它的优点是语义简单，适合作为 correctness reference（正确性参考）。

在 reference 阶段，`dense_decode_attention` 是 FP32 ground truth（FP32 真值）：

```text
对每个 batch b：
    只看 context_lens[b] 以内的 token
    q[b] 和 k[b, :context_lens[b]] 做 dot product
    对 token 维度 softmax
    用 softmax 权重加权 v[b, :context_lens[b]]
```

## 9. Paged KV Layout

Paged KV layout（分页 KV 布局）把 KV cache 切成固定大小的 block（块），例如每个 block 存 16 个 token：

```text
k_cache: [num_blocks, block_size, num_heads, head_dim]
v_cache: [num_blocks, block_size, num_heads, head_dim]
```

逻辑 token 到物理 cache 的映射由 block table（块表）完成：

```text
physical_block = block_tables[b, t // block_size]
slot           = t % block_size
```

然后读取：

```text
k_cache[physical_block, slot, h, d]
v_cache[physical_block, slot, h, d]
```

Paged KV 的主要收益是：

```text
1. 减少不同请求长度带来的 padding 浪费。
2. 减少动态请求进入/结束造成的显存碎片。
3. 更容易复用释放后的 KV block。
4. 更适合 LLM serving（LLM 服务推理）的动态 workload（负载）。
```

它不是压缩 KV，也不会让有效历史 token 数减少。如果上下文长度是 `L`，decode attention 仍然要读取大约 `L` 个 token 的 K/V。

## 10. Reference.py 的作用

`src/paged_kv_attention/reference.py` 不是最终高性能实现，而是标准答案程序。

它包含两层：

```text
dense_decode_attention:
    连续 KV layout 下的数学标准答案。

paged_decode_attention:
    分页 KV layout 下的语义标准答案。
```

两者的 attention 数学相同：

```text
scores = q @ k^T / sqrt(head_dim)
probs  = softmax(scores)
out    = probs @ v
```

区别只在于 K/V 的读取地址：

```text
dense:
    k[b, t, h, d]
    v[b, t, h, d]

paged:
    block_id = block_tables[b, t // block_size]
    slot     = t % block_size
    k_cache[block_id, slot, h, d]
    v_cache[block_id, slot, h, d]
```

reference 阶段的目标是证明：

```text
paged reference 输出 ≈ dense reference 输出
```

Week 2-3 写 Triton kernel（Triton 内核）时，再证明：

```text
Triton paged attention kernel 输出 ≈ paged reference 输出 ≈ dense reference 输出
```

## 11. 当前项目真正优化什么

这个项目围绕 decode 阶段的 KV cache 做两类优化：

```text
1. 显存组织方式：
   用 paged KV cache 管理不同长度、动态进入和结束的请求。

2. Kernel 读取性能：
   在 paged layout 下正确且高效地读取 K/V，完成 softmax 和加权 V。
```

reference 阶段不追求性能，重点是定义 correctness（正确性）：

```text
哪些 token 能读？
哪些 slot 绝对不能读？
分页地址怎么从逻辑 token 映射到物理 block？
最终输出应该和 dense reference 如何对齐？
```

这些定义清楚后，后续 kernel 优化才有可靠的裁判。
