# Resume Snippets

按项目实际完成状态选用对应版本。三条纪律：

- 不写“复刻 vLLM / 高性能 serving engine”，定位是 PagedAttention-inspired decode attention kernel lab；
- 性能数字必须绑定具体硬件与配置（GPU 型号、dtype、shape），下文的 RTX 4090 按实际租用硬件替换；
- 只声称实际交付的内容——没做 CUDA extension 就不能出现 CUDA extension。

## Performance Checkpoint 版本

### 中文

> 独立实现 LLM 推理 decode 阶段的 Paged-KV Attention Triton 算子，支持 variable-length batch 与 block-table KV cache layout；以 FP32 reference 为基准构建覆盖末块未满、乱序 block table 等边界的 correctness 测试，并对比 PyTorch SDPA 与 FlashInfer。通过 program saturation 与有效带宽分析定位小 batch 长 context 的并行度瓶颈：`B=1,S=16K` 时 single-pass Paged Triton 相对 FlashInfer 慢 `9.22x`，到 `B=16` 基本收敛（RTX 5090 / FP16）。

### English

> Built a paged-KV attention Triton kernel for LLM decode inference with variable-length batching and block-table KV cache layout; developed FP32-reference correctness tests for partial-block and shuffled-table cases, and benchmarked against PyTorch SDPA and FlashInfer. Program-saturation and effective-bandwidth analysis exposed a `9.22x` single-pass gap to FlashInfer at `B=1,S=16K`, which disappeared near `B=16` (RTX 5090 / FP16).

## Split-KV Checkpoint 版本

### 中文

> 独立实现面向 LLM decode 阶段的 Paged-KV Attention Triton 算子，支持 variable-length batch 与 block-table KV cache layout；针对小 batch 长 context 下 program 数不足的问题，实现 Flash-Decoding 风格 partial/reduce split-KV 与 adaptive dispatch，并量化优化前后延迟、有效带宽和大 batch 收益边界（RTX 5090 / FP16）。

### English

> Built a paged-KV attention Triton kernel for LLM decode inference with variable-length batching and block-table KV layout; implemented Flash-Decoding-style partial/reduce split-KV with adaptive dispatch to address insufficient program-level parallelism at small batch and long context, quantifying latency, effective-bandwidth gains, and the large-batch break-even boundary (RTX 5090 / FP16).

## Final Checkpoint 版本（Triton Split-KV + 最小 CUDA Port）

### 中文

> 独立实现面向 LLM decode 阶段的 Paged-KV Attention 自定义算子：在 Triton 中实现 adaptive split-KV，解决小 batch 长 context 的并行度不足；将已验证的 single-pass paged kernel 移植为限定范围的 CUDA/C++ PyTorch extension，复用 correctness tests，并量化 Triton/CUDA 的接口、延迟、带宽与工程成本差异（RTX 5090 / FP16）。

### English

> Built a custom paged-KV attention operator for LLM decode inference: implemented adaptive split-KV in Triton to address insufficient small-batch, long-context parallelism, then ported the validated single-pass paged kernel to a scoped CUDA/C++ PyTorch extension; reused the same correctness suite and quantified Triton/CUDA differences in interface, latency, bandwidth, and engineering cost (RTX 5090 / FP16).

## 面试话术备忘

- 被问“为什么 CUDA 没重复实现 split-KV”：说明 CUDA 阶段目标是理解手动线程映射与归约，主线范围锁死 single-pass；CUDA split-KV 是 stretch goal。
- 被问“用 CUDA 你会怎么写”：答案在 `docs/cuda-design-sketch.md`（线程块映射、shared memory staging、warp 归约、向量化加载）。
- 被问“和 vLLM / FlashInfer 差在哪”：说明接口和生产特性差异；再展示 CUDA 13 定量结果，解释
  FlashInfer 的 split-KV 在小 batch 长 context 下提供 context parallelism，而大 batch 时各路径
  收敛到带宽平台。
- 被问“哪个 bug 印象最深”：从 lab notes 里挑亲手 debug 的案例（mask 边界 / tolerance / 越界读）。
