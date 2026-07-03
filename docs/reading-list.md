# Reading List

分三层：**Tier 1 必读**（Week 0，timebox 1.5 天）、**Tier 2 按周按需**（做到哪周读哪份）、**Tier 3 明确不读**（暑期内）。

读法纪律：每份材料带着"要回答的问题"去读，单份 timebox 60-90 分钟，读完写 3-5 行笔记进 lab notes。读不懂的部分标记后跳过，做到对应周自然会懂。

## Tier 1: 必读（Week 0，共约 1.5 天）

### 1. PagedAttention 论文（SOSP 2023）

- 链接：https://arxiv.org/abs/2309.06180
- 读什么：KV cache 内存问题的定义（fragmentation / 动态增长）、block + block table 设计、block size trade-off 讨论、evaluation 里的 memory waste 分析。
- 跳过：分布式执行、scheduler 细节、beam search 相关。
- 读完要能回答：为什么连续分配 KV cache 会浪费显存？block table 如何把 logical position 映射到 physical block？block size 太大/太小分别有什么代价？

### 2. Hugging Face paged attention / continuous batching 文档

- 链接：https://huggingface.co/docs/transformers 内搜 "paged attention" / "continuous batching"
- 读什么：decode path 的 `q_len=1` 语义、`block_table` 的 shape `(batch_size, max_blocks_per_seq)`、`cache_seqlens` 的含义。
- 读完要能回答：本项目 `paged_attention(q, k_cache, v_cache, block_tables, seq_lens)` 的每个参数为什么长这个形状？（这份文档几乎就是 Level 1 接口的参照答案。）

### 3. Triton 官方教程 01 / 02 / 03

- 链接：https://triton-lang.org/main/getting-started/tutorials/
- 01 vector addition：program id、`tl.load`/`tl.store`、mask 语义；
- 02 fused softmax：**最重要**，它是本项目 kernel 的直接前身——row-wise 归约、数值稳定的 max 减法；
- 03 matrix multiplication：分块循环、accumulator 模式（online softmax 的循环结构与它同构）。
- 读完要能做：不看教程，独立写出并跑通一个带 mask 的 row-wise softmax kernel。

## Tier 2: 按周按需

### Week 2-3（写 kernel 时）

- **FlashAttention-2 论文**：https://arxiv.org/abs/2307.08691 ——只读分块算法与 online softmax 部分（§2-3），目标是能手推 running max / running sum 的更新式；warp 划分细节跳过。
- **Triton 教程 06 fused attention**：作为代码参照，但注意它是 prefill 场景（q_len 大、Q 也分块），decode 的 `q_len=1` 结构更简单，不要照抄。

### Week 4（benchmark / profiling 时）

- **FlashInfer 文档**：https://docs.flashinfer.ai ——`BatchDecodeWithPagedKVCacheWrapper` API 与 KV layout 章节，接 baseline 用；顺手记录它的接口和你的接口差异（写进 design doc，面试素材）。
- **NCU 权限问题官方页**：https://developer.nvidia.com/ERR_NVGPUCTRPERM ——Week 0 验证失败时按此排查/确认 fallback。
- Nsight Compute / Nsight Systems 文档：只查用到的功能，不通读。

### Week 5（按门控选择读对应一份）

- **选 split-KV（默认）**：PyTorch Flash-Decoding 博客 https://pytorch.org/blog/flash-decoding/ ——partial softmax 并行 + reduce 的核心思路；可对照 vLLM PagedAttention V2 的 split 设计。
- **选 CUDA port**：PyTorch C++/CUDA extension 官方教程 https://pytorch.org/tutorials/advanced/cpp_extension.html + vLLM `csrc/attention/attention_kernels.cu` 源码（只读 decode kernel 主循环，理解线程映射即可）。
- **选 allocator**：vLLM KV cache manager 相关设计文档与源码（`docs.vllm.ai` design 章节）。

### Week 6（写 limitations 时）

- **vAttention 论文**：https://arxiv.org/abs/2405.04437 ——只读 intro 和对 PagedAttention 的批评部分（非连续 virtual layout 的编程与性能代价），作为报告里的反方证据。PagedAttention 不是唯一解，报告承认这一点反而更可信。

## Tier 3: 明确不读（暑期内）

以下材料有价值但不在 6 周预算内，知道一句话结论即可，用到再查：

- Orca 论文全文（一句话：iteration-level scheduling 是 continuous batching 的来源）；
- DeepSpeed-FastGen / Dynamic SplitFuse；
- SGLang RadixAttention（一句话：prefix 复用是 KV cache 的另一维度）；
- TensorRT-LLM 全套文档（只在需要引用 GQA/MQA 或 KV manager 佐证时查对应页）；
- 任何 INT8/FP8 KV cache 材料。
