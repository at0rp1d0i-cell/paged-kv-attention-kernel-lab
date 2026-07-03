# Lab Notes: Week 0

## 1. 本周目标

验证 GPU / Triton / CUDA extension / profiling 工具链，完成 repo 工程化，并读完 Week 1-3 必需材料。

## 2. Reading Notes

### PagedAttention

- TODO: 连续 KV cache 浪费显存的机制。
- TODO: block table 的 logical-to-physical 映射。
- TODO: block size trade-off。

### Hugging Face paged attention / continuous batching

- TODO: `q_len=1` decode 语义。
- TODO: `block_tables` shape 与含义。
- TODO: `cache_seqlens` / `seq_lens` 的含义。

### Triton Tutorials 01 / 02 / 03

- TODO: `program_id` 映射方式。
- TODO: mask load/store 语义。
- TODO: row-wise softmax 的 max / sum reduction。
- TODO: matmul block loop 与 online softmax 循环结构的相似点。

## 3. Toolchain Notes

- TODO: PyTorch + Triton 版本组合。
- TODO: GPU smoke 结果。
- TODO: CUDA extension compile 结果。
- TODO: NCU 权限结果。

## 4. 最难 / 最烦的点

TODO

## 5. 学到什么

TODO

## 6. 享受 / 排斥

区分“工具链痛苦”和“方向排斥”：

- 享受：
  - TODO
- 排斥：
  - TODO
- 更像工具链问题：
  - TODO
- 更像方向不适：
  - TODO

## 7. Week 1 入口判断

- [ ] 环境组合已记录。
- [ ] NCU 或 fallback 已判定。
- [ ] 三份必读材料有简短笔记。
- [ ] 能解释 Week 1 接口和 block table 数据结构。
- [ ] 准备开始 dense reference / paged reference / block-table generator。
