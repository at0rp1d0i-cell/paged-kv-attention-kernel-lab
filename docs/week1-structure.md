# Week 1 Structure

Week 1 的目标是先把 reference（参考实现）和 paged layout（分页布局）讲清楚，再进入
Triton kernel（Triton 内核）。本周不追求性能，正确性标准是和 FP32 dense reference
对齐。

## 模块边界

- `src/paged_kv_attention/layouts.py`: 记录 Week 1 tensor layout（张量布局）和静态维度约束。
- `src/paged_kv_attention/reference.py`: 放 dense reference 与 paged reference。
- `src/paged_kv_attention/block_table.py`: 放 block table（块表）相关 helper（辅助函数）和 generator（生成器）。
- `tests/test_reference_contracts.py`: reference 阶段的接口和 helper contract（契约）测试。

## 推荐实现顺序

1. `dense_decode_attention`
   - 输入：`q [B, H, D]`, `k/v [B, S, H, D]`, `context_lens [B]`
   - 输出：`out [B, H, D]`, dtype 为 `torch.float32`
   - 正确性：每个 batch 只看 `context_lens[b]` 以内的 token。
2. `make_random_block_tables`
   - 输入：`context_lens`, `block_size`, optional seed
   - 输出：`block_tables [B, max_blocks_per_seq]`, `num_physical_blocks`
   - 正确性：尽量生成 random-order（随机顺序）和 non-contiguous（非连续）映射。
3. `paged_decode_attention`
   - 输入：`q`, `k_cache/v_cache [num_blocks, block_size, H, D]`, `block_tables`, `context_lens`
   - 输出：和 dense reference 对齐的 FP32 tensor
   - 正确性：只读有效 token，不读最后 block 的未使用 slot。

## Week 1 边界用例

- `batch=1` 和 multi-batch（多 batch）。
- `context_len < block_size`。
- `context_len == block_size`。
- `context_len > block_size`。
- 最后一个 block 未填满。
- block table 非连续、随机顺序。
- 未用 slot 填 garbage values（垃圾值）后结果不变。

## 学习检查点

- 能解释 dense attention 的 score / softmax / value 加权过程。
- 能手推 token index 到 paged cache 地址：
  `physical_block = block_tables[b, t // block_size]`,
  `slot = t % block_size`。
- 能说明为什么 unused slots 填垃圾值可以捕捉越界读取。
