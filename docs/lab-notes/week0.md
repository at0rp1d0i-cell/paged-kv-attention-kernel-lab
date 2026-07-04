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

- PyTorch + Triton 版本组合：`torch==2.8.0+cu128`、`triton==3.4.0`、Python `3.12.3`，由 `uv.lock` 固定。
- GPU smoke 结果：PyTorch CUDA tensor op、Triton vector add、CUDA extension compile/run 均通过。
- CUDA extension compile 结果：`scripts/gpu_smoke.py` 会在未设置 `TORCH_CUDA_ARCH_LIST` 时自动使用当前 GPU capability（RTX 5090 上为 `12.0`）。
- NCU 权限结果：`ncu` 已安装，但容器内 probe 返回 `ERR_NVGPUCTRPERM`，`/proc/driver/nvidia/params` 显示 `RmProfilingAdminOnly: 1`。
- Profiling fallback：容器权限内不继续尝试修 NCU；后续默认用 CUDA events、`torch.profiler` 和 analytical bandwidth model。

## 4. 最难 / 最烦的点

工具链最烦的点是 `uv.lock` 的 registry 与临时镜像源不一致，以及 `ncu` counters 受宿主机 driver 权限限制。结论是把清华源写进运行命令，用 `uv lock` 重新固定依赖；把 NCU 权限问题记录为非阻塞 fallback，而不是在容器内继续消耗时间。

## 5. 学到什么

TODO

## 6. 享受 / 排斥

区分“工具链痛苦”和“方向排斥”：

- 享受：
  - TODO
- 排斥：
  - TODO
- 更像工具链问题：
  - `ERR_NVGPUCTRPERM` 是宿主机 driver / container capability 问题，不是 kernel 实现问题。
- 更像方向不适：
  - TODO

## 7. Week 1 入口判断

- [x] 环境组合已记录。
- [x] NCU 或 fallback 已判定。
- [ ] 三份必读材料有简短笔记。
- [ ] 能解释 Week 1 接口和 block table 数据结构。
- [ ] 准备开始 dense reference / paged reference / block-table generator。
