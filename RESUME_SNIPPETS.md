# Resume Snippets

按项目实际完成状态选用对应版本。三条纪律：

- 不写“复刻 vLLM / 高性能 serving engine”，定位是 PagedAttention-inspired decode attention kernel lab；
- 性能数字必须绑定具体硬件与配置（GPU 型号、dtype、shape），下文的 RTX 4090 按实际租用硬件替换；
- 只声称实际交付的内容——没做 CUDA extension 就不能出现 CUDA extension。

## 检查点 1 版本（Week 4 末，Level 1 完成即可投）

### 中文

> 独立实现 LLM 推理 decode 阶段的 Paged-KV Attention Triton 算子，支持 variable-length batch 与 block-table KV cache layout；以 FP32 reference 为基准构建覆盖末块未满、乱序 block table 等边界的 correctness 测试，以 PyTorch SDPA 与 FlashInfer 为 baseline 完成系统化 benchmark，并用有效带宽利用率分析论证 decode attention 的 memory-bound 瓶颈（RTX 4090 / FP16）。

### English

> Built a paged-KV attention Triton kernel for LLM decode inference with variable-length batching and block-table KV cache layout; developed FP32-reference correctness tests covering partial-block and shuffled-block-table edge cases, benchmarked against PyTorch SDPA and FlashInfer baselines, and used effective-bandwidth-utilization analysis to characterize the memory-bound bottleneck (RTX 4090 / FP16).

## 版本 A: 最终形态，Triton 主线 + split-KV（无 CUDA extension）

### 中文

> 独立实现面向 LLM decode 阶段的 Paged-KV Attention Triton 算子，支持 variable-length batch、block-table KV cache layout 与 GQA；针对小 batch 长 context 下 SM 占用不足的问题，实现 Flash-Decoding 风格 split-KV 并量化前后收益；以 PyTorch SDPA 与 FlashInfer 为 baseline 构建 correctness / benchmark / profiling 证据链，用带宽利用率解释 memory-bound 边界与优化天花板（RTX 4090 / FP16）。

### English

> Built a paged-KV attention Triton kernel for LLM decode inference (variable-length batching, block-table KV layout, GQA); implemented Flash-Decoding-style split-KV partitioning to address low SM occupancy at small batch and long context, with quantified before/after gains; established a correctness/benchmark/profiling evidence chain against PyTorch SDPA and FlashInfer baselines, using bandwidth-utilization analysis to explain the memory-bound performance envelope (RTX 4090 / FP16).

## 版本 B: 最终形态，门开后（Triton + 最小 CUDA extension）

### 中文

> 独立实现面向 LLM decode 阶段的 Paged-KV Attention 自定义算子，完成 Triton kernel 与限定范围的 CUDA/C++ PyTorch extension 两套实现并做接口与性能对照；支持 variable-length batch 与 block-table KV cache layout，以 PyTorch SDPA 与 FlashInfer 为 baseline 构建 correctness / benchmark / profiling 证据链，用带宽利用率分析 memory-bound 瓶颈（RTX 4090 / FP16）。

### English

> Built a custom paged-KV attention operator for LLM decode inference in both Triton and a scoped CUDA/C++ PyTorch extension, with cross-implementation interface and performance comparison; supported variable-length batching and block-table KV cache layout, and established a correctness/benchmark/profiling evidence chain against PyTorch SDPA and FlashInfer baselines with bandwidth-utilization analysis (RTX 4090 / FP16).

## 版本 C: Week 5 选了 allocator

在版本 A 基础上，把 split-KV 一句替换为：

### 中文

> 实现 mini KV block allocator 与 request 到达 / 结束模拟，量化分页策略对 fragmentation、block 复用与显存占用的影响。

### English

> implemented a mini KV block allocator with request arrival/finish simulation, quantifying how paging policy affects fragmentation, block reuse, and memory footprint.

## 面试话术备忘

- 被问“用 CUDA 你会怎么写”：答案在 `docs/cuda-design-sketch.md`（线程块映射、shared memory staging、warp 归约、向量化加载）。
- 被问“和 vLLM / FlashInfer 差在哪”：引用 benchmark 中与 FlashInfer 的实测差距 + profiling report 的归因，不空谈。
- 被问“哪个 bug 印象最深”：从 lab notes 里挑亲手 debug 的案例（mask 边界 / tolerance / 越界读）。
