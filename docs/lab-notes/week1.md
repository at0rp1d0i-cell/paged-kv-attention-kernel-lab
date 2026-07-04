# Week 1 Lab Notes

## 本周目标

- 实现 FP32 dense decode attention reference。
- 定义 paged KV cache layout 与 block-table semantics。
- 让 paged reference 与 dense reference 在边界 case 上对齐。

## 最难的 bug

TODO: 记录一个你亲手 debug 的 correctness bug，例如 mask 边界、block table 下标或 dtype
tolerance（容差）。

## 学到的内容

TODO: 记录你对 online softmax、paged layout 或 reference testing 的理解变化。

## 享受 / 排斥

TODO: 区分工具链痛苦和方向本身的排斥感。这个判断会服务 Week 4 末 CUDA gate（门控）。

## 下一步

TODO: 写下 Week 2-3 进入 Triton kernel 前还不确定的问题。
