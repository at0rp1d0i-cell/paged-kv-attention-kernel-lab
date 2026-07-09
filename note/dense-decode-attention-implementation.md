# Reference 阶段实现笔记：dense_decode_attention

这份笔记记录 `src/paged_kv_attention/reference.py` 里 `dense_decode_attention` 的实现思路。它属于实现笔记；attention（注意力）、prefill（预填充）、decode（解码）和 KV cache（KV 缓存）的前置知识见 `note/reference-stage-attention-kv-cache.md`。

## 1. 函数定位

`dense_decode_attention` 是 reference 阶段的第一个核心 reference（参考实现）。它的目标不是性能，而是用最清楚的 PyTorch 逻辑定义 decode attention（解码注意力）的正确结果。

后续关系是：

```text
dense_decode_attention
    ↓ 定义 FP32 ground truth（真值）

paged_decode_attention
    ↓ 对齐 dense reference，并定义 paged layout（分页布局）语义

Triton paged attention kernel
    ↓ 对齐 paged reference，并追求性能
```

## 2. 输入输出约定

函数输入：

```text
q:            [B, H, D]
k:            [B, S, H, D]
v:            [B, S, H, D]
context_lens: [B]
```

其中：

```text
B = batch size（批大小）
H = num_heads（注意力头数量）
D = head_dim（每个 head 的维度）
S = max_context_len（最大上下文长度）
```

输出：

```text
out: [B, H, D]
```

## 3. 先做 correctness checks

实现第一步不是直接算 attention，而是先确认输入 layout（布局）和有效 token 范围是合法的。

需要检查：

```text
q.ndim == 3
k.ndim == 4
v.ndim == 4
k 的 [B, H, D] 和 q 对齐
v 的 shape 和 k 完全一致
context_lens 是 [B]
context_lens 非负
context_lens 不能超过 S
```

这样做的原因是：reference 是后续 paged reference（分页参考实现）和 Triton kernel（Triton 内核）的 ground truth。如果 reference 接受了错误 shape，后面测试失败时就很难判断是测试数据错、reference 错，还是 kernel 错。

## 4. 用 FP32 做标准答案

reference 阶段的 dense reference 默认作为 FP32 ground truth：

```python
q_f = q.to(torch.float32)
k_f = k.to(torch.float32)
v_f = v.to(torch.float32)
```

这样即使后面 kernel 用 FP16/BF16，也可以用 FP32 reference 判断误差是否在 tolerance（容差）范围内。

输出也明确分配成 FP32：

```python
out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
```

## 5. 每个 batch item 单独计算

decode attention 里，每个 batch item 的有效上下文长度可能不同，所以按 batch 循环最直观：

```python
for b in range(B):
    valid_len = int(context_lens[b].item())
```

这里：

```text
context_lens[b]:        取出第 b 个 batch 的有效上下文长度，是 0-d tensor
context_lens[b].item(): 转成 Python 标量
int(...):               确保后面可以作为切片长度使用
```

然后只取有效 token：

```python
q_b = q_f[b]              # [H, D]
k_b = k_f[b, :valid_len]  # [T, H, D], T = valid_len
v_b = v_f[b, :valid_len]  # [T, H, D]
```

这一步实现了 mask（掩码）语义：`t >= context_lens[b]` 的 token 根本不会进入计算。

## 6. 计算 attention scores

数学目标：

```text
scores[h, t] = dot(q_b[h], k_b[t, h]) * scale
```

代码：

```python
scores = (k_b * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale
```

shape trace（形状追踪）：

```text
q_b:              [H, D]
q_b.unsqueeze(0): [1, H, D]
k_b:              [T, H, D]
相乘后:           [T, H, D]
sum(dim=-1):      [T, H]
transpose(0, 1):  [H, T]
scores:           [H, T]
```

`scale` 默认是：

```text
1 / sqrt(head_dim)
```

也就是：

```python
default_attention_scale(D)
```

它的作用是避免 `head_dim` 变大时 dot product 数值过大，导致 softmax 过早变得极尖锐。

## 7. 对 token 维度做 softmax

`scores` 的 shape 是：

```text
[H, T]
```

其中最后一维 `T` 是 token/context 维度，所以 softmax 要写：

```python
probs = torch.softmax(scores, dim=-1)
```

语义是：

```text
probs[h, :] = softmax(scores[h, :])
```

也就是每个 head 独立地对所有有效历史 token 做 softmax。

## 8. 用 probs 加权 V

当前：

```text
probs: [H, T]
v_b:   [T, H, D]
```

先把 `v_b` 转成按 head 排列：

```python
v_by_head = v_b.transpose(0, 1)
```

shape：

```text
v_b:       [T, H, D]
v_by_head: [H, T, D]
```

然后加权求和：

```python
out[b] = (probs.unsqueeze(-1) * v_by_head).sum(dim=1)
```

shape trace：

```text
probs.unsqueeze(-1): [H, T, 1]
v_by_head:           [H, T, D]
相乘后:              [H, T, D]
sum(dim=1):          [H, D]
out[b]:              [H, D]
```

这对应公式：

```text
out[b, h] = sum_t probs[h, t] * v_b[t, h]
```

## 9. 当前验证状态和下一步

当前 scaffold tests（脚手架测试）已经验证：

```text
输出 shape == q.shape
输出 dtype == torch.float32
```

但这还不够。下一步应该补 dense reference 的数值正确性测试：

```text
1. 用 B=1, H=1, D=2, S=2 的小 case 验证 softmax(q @ k^T) @ v。
2. 用 context_lens < S 的 case 验证无效 token 不会影响输出。
```

这样 dense reference 才真正可以作为后续 paged reference 和 Triton kernel 的裁判。
