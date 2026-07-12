# PyTorch 常用语法速查

这份笔记只收录 paged KV attention 项目中已经使用或近期会使用的 PyTorch 语法。重点不是
记忆 API，而是理解每个操作如何改变 shape（形状）、dtype（数据类型）和 device（设备）。

## 1. 创建 Tensor

从已有数据创建：

```python
x = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=torch.float32,
    device="cuda",
)
```

常用构造函数：

```python
torch.empty((B, H, D), device=device, dtype=torch.float32)
torch.zeros((H, D), device=device, dtype=torch.float32)
torch.full((H,), -torch.inf, device=device, dtype=torch.float32)
torch.randn((B, S, H, D), device=device, dtype=torch.float16)
torch.arange(S, device=device)
```

跟随已有 tensor 的 shape/device：

```python
out = torch.empty_like(q, dtype=torch.float32)
v = torch.randn_like(k)
mask = torch.zeros_like(context_lens, dtype=torch.bool)
```

常见误区：`torch.empty` 不会初始化数据，必须保证所有输出位置之后都会被写入。

## 2. 查看 Shape、Dtype 与 Device

```python
B, H, D = q.shape

q.ndim
q.dtype
q.device
q.is_cuda
q.is_contiguous()
q.numel()
```

本项目 wrapper（封装函数）通常先验证这些 contract（契约），再 launch Triton kernel。

## 3. 基础索引与切片

假设：

```text
k: [B, S, H, D]
```

```python
k[b]                  # [S, H, D]
k[b, :valid_len]      # [T, H, D]
k[b, t, h]            # [D]
k[b, t, h, d]         # scalar
k[:, :, h, :]         # [B, S, D]
```

布尔索引：

```python
used_block_ids = block_tables[block_tables >= 0]
```

注意：高级索引和布尔索引可能产生新 tensor；不要默认它们只是零成本 view（视图）。

## 4. 增加与删除维度

`unsqueeze` 在指定位置增加大小为 1 的维度：

```python
q_b:                 [H, D]
q_b.unsqueeze(0):    [1, H, D]

probs:               [H, T]
probs.unsqueeze(-1): [H, T, 1]
```

等价索引写法：

```python
x.unsqueeze(0)   == x[None, ...]
x.unsqueeze(-1)  == x[..., None]
```

删除大小为 1 的维度：

```python
x.squeeze(-1)
```

不要对不确定大小的维度随意使用无参数 `squeeze()`，否则 batch size 为 1 时可能误删 batch
维度。

## 5. 转置与重排维度

交换两个维度：

```python
v_tile:                    [T, H, D]
v_tile.transpose(0, 1):    [H, T, D]
```

任意重排：

```python
x.permute(0, 2, 1, 3)
```

`transpose/permute` 通常返回 view，结果可能 non-contiguous（非连续）。如果后续接口要求连续
内存，需要显式：

```python
x = x.transpose(0, 1).contiguous()
```

## 6. Broadcasting

broadcasting（广播）从末尾维度开始匹配；对应维度必须相等，或其中一个为 1。

```text
k_tile:             [T, H, D]
q_b.unsqueeze(0):   [1, H, D]
相乘结果:            [T, H, D]
```

另一个常见例子：

```text
p_tile.unsqueeze(-1): [H, T, 1]
v_tile_by_head:        [H, T, D]
相乘结果:               [H, T, D]
```

广播通常不需要真正复制数据，但后续操作仍可能产生完整输出 tensor。

## 7. Reduction

reduction（归约）沿指定维度聚合数据：

```python
x.sum(dim=-1)
x.max(dim=-1).values
x.mean(dim=0)
torch.any(mask)
torch.all(mask)
```

shape 示例：

```text
x:                 [T, H, D]
x.sum(dim=-1):     [T, H]
x.sum(dim=0):      [H, D]
x.max(dim=1).values: [T, D]
```

`dim=-1` 表示最后一个维度。`keepdim=True` 可以保留被归约的维度：

```python
x.sum(dim=-1, keepdim=True)  # [T, H, 1]
```

## 8. Element-wise 操作

```python
torch.exp(x)
torch.maximum(a, b)
torch.where(mask, a, b)
x * y
x + y
x / y
```

`torch.maximum` 是逐元素比较：

```python
torch.maximum(
    torch.tensor([2, 8, 3]),
    torch.tensor([5, 1, 3]),
)
# tensor([5, 8, 3])
```

它和 `x.max(dim=...)` 不同：后者是在一个 tensor 内沿某个维度做归约。

## 9. Softmax 与 Attention Shape

```python
scores = (k_b * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale
probs = torch.softmax(scores, dim=-1)
out = (probs.unsqueeze(-1) * v_by_head).sum(dim=1)
```

shape trace：

```text
k_b:                     [T, H, D]
q_b.unsqueeze(0):        [1, H, D]
乘法后:                   [T, H, D]
sum(dim=-1):             [T, H]
transpose(0, 1):         [H, T]
softmax(dim=-1):         [H, T]
probs.unsqueeze(-1):     [H, T, 1]
v_by_head:               [H, T, D]
sum(dim=1):              [H, D]
```

## 10. Dtype 与 Device 转换

```python
x_f32 = x.to(torch.float32)
x_cuda = x.to("cuda")
x_int = x.to(dtype=torch.int32, device="cuda")
```

项目中的 reference 通常把输入转成 FP32：

```python
q_f = q.to(torch.float32)
k_f = k.to(torch.float32)
v_f = v.to(torch.float32)
```

不要无意间把 CUDA tensor 转回 CPU；device 不同的 tensor 不能直接参与普通运算。

## 11. 整数除法与余数

分页索引：

```python
logical_block = token_idx // block_size
slot = token_idx % block_size
```

tensor 版本：

```python
required_blocks = torch.div(
    context_lens + block_size - 1,
    block_size,
    rounding_mode="floor",
)
```

向上取整除法：

```text
ceil(n / d) = floor((n + d - 1) / d)
```

## 12. Stack、Concatenate 与 View

将多个同 shape tensor 增加一个新维度后堆叠：

```python
k_b = torch.stack(k_tokens, dim=0)  # 多个 [H,D] -> [T,H,D]
```

沿已有维度拼接：

```python
torch.cat([a, b], dim=0)
```

改变 shape：

```python
x.reshape(B, H, D)
x.view(B, H, D)
```

`view` 要求 stride（步长）兼容；不确定时优先使用 `reshape`，但要意识到它可能复制数据。

## 13. 随机数与可复现测试

```python
generator = torch.Generator(device="cuda").manual_seed(31)
x = torch.randn(
    shape,
    generator=generator,
    device="cuda",
    dtype=torch.float16,
)
```

测试中固定 seed 可以复现失败输入，但性能 benchmark 不应在计时区间内生成随机数。

## 14. Correctness Test

数值对齐：

```python
torch.testing.assert_close(
    actual,
    expected,
    atol=2e-3,
    rtol=2e-3,
)
```

近似判断使用：

```text
|actual - expected| <= atol + rtol * |expected|
```

测试不同 dtype 时必须明确 tolerance（容差），不能只写“看起来差不多”。

## 15. CUDA 同步

GPU 操作通常是 asynchronous（异步）的。需要等待 GPU 完成时：

```python
torch.cuda.synchronize()
```

correctness 测试中的 tensor 比较通常会触发必要同步；benchmark 应使用 CUDA events，不能
只用普通 CPU wall-clock 包围异步 kernel launch。

## 16. 本项目常用 Shape 模板

```text
decode Q:        [B, H, D]
dense K/V:       [B, S, H, D]
paged K/V:       [num_blocks, block_size, H, D]
block_tables:    [B, max_blocks_per_seq]
context_lens:    [B]
output:          [B, H, D]
```

遇到 PyTorch 语法不确定时，先写出操作前后的 shape，再决定 `dim`、`unsqueeze` 或
`transpose`，比直接试 API 更可靠。
