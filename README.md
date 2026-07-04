# Paged-KV Attention Kernel Lab

> 暑期挑战项目提案：面向 LLM 推理 decode 阶段的 Paged KV Cache Attention 自定义算子、benchmark 与 profiling 实验室。

## Quickstart

本项目统一使用 `uv` 管理 Python 环境。当前机器普通 PyPI 包使用清华镜像，PyTorch CUDA wheel 固定走 `pyproject.toml` 中的 cu128 PyTorch index。

```bash
cd /root/paged-kv-attention-kernel-lab
python -m pip install -U uv
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=600 uv sync --locked --group dev
bash scripts/check_env.sh
bash scripts/run_tests.sh
uv run python scripts/gpu_smoke.py
```

当前容器内 `ncu` 能启动但不能读取 NVIDIA GPU performance counters（性能计数器）：`/proc/driver/nvidia/params` 显示 `RmProfilingAdminOnly: 1`，`ncu` probe 会返回 `ERR_NVGPUCTRPERM`。这不是项目阻塞项；在只有容器内权限时，profiling（性能剖析）默认 fallback（回退）到 CUDA events（CUDA 事件）测 latency（延迟）、`torch.profiler` 看 operator timeline（算子时间线）、analytical bandwidth model（解析带宽模型）估算 effective bandwidth（有效带宽）。

## 1. 项目定位

这个项目是一个围绕真实 LLM inference infra 痛点设计的个人工程项目。

目标是从一个具体系统问题出发：

> 在线 LLM 推理 decode 阶段，每个请求都要反复读取历史 KV cache；当 batch size、context length 和并发请求数上升时，KV cache 的显存占用、带宽压力、碎片管理和不规则访存会成为服务吞吐和延迟的核心瓶颈。

项目最终交付一个独立 repo，包含：

- PyTorch reference implementation
- Triton paged attention kernel
- CUDA/C++ PyTorch extension（Week 4 末门控通过后才做，详见第 4 节）
- correctness tests
- benchmark harness
- profiler 分析报告
- 技术 README / report

这个项目服务于简历主线：

- CUDA
- 算子开发
- PyTorch custom op
- LLM inference
- AI infra
- performance benchmark
- profiler-driven optimization

## 2. 为什么这个问题是真痛点

### 2.1 Decode 阶段不是单纯 GEMM 问题

LLM 推理通常可以拆成两个阶段：

- prefill：一次性处理 prompt，计算量大，GEMM-heavy；
- decode：每次生成一个 token，`q_len = 1`，但要读取该请求历史上所有 token 的 K/V。

decode attention 的核心成本不是公式本身，而是持续读取 KV cache：

```text
softmax(Q K^T / sqrt(d)) V
```

当 `q_len = 1` 时，Q 很小，K/V 很大。长上下文和高并发下，decode 阶段很容易变成 memory-bandwidth-bound（显存带宽受限）。

### 2.2 KV cache 是在线推理的动态显存大户

模型权重是固定成本，KV cache 是随请求数量、上下文长度、batching 策略动态增长的成本。

在线 serving 中，请求具有以下特点：

- 到达时间不同；
- prompt 长度不同；
- decode 长度不同；
- 结束时间不同；
- 有的请求很短，有的请求很长。

如果为每个请求连续分配一大段 KV cache，会带来：

- 显存浪费；
- 内存碎片；
- 长短请求混合时调度困难；
- batch 重组成本高。

Paged KV cache 的思想是把每个请求的 KV cache 切成 fixed-size blocks，通过 block table 做逻辑 token position 到物理 block 的映射。这类似操作系统分页内存管理。

### 2.3 系统层和算子层存在天然冲突

LLM serving 系统希望：

- continuous batching；
- request-level 动态插入和退出；
- prefix cache 复用；
- KV block 复用；
- 高并发下显存利用率高。

但底层 kernel 希望：

- 连续访存；
- 规则 shape；
- coalesced memory access；
- warp 内线程访问模式一致；
- 尽量减少间接寻址和 branch。

Paged-KV attention 刚好站在这个冲突点上：

> 系统层需要灵活的 KV block 管理，算子层需要高效读取这些被分页管理的 K/V。

这也是它比普通 CUDA 练习更有价值的原因。

## 3. 为什么这个问题有技术难度

### 3.1 Paged layout 带来间接寻址

普通连续 KV layout 可以直接通过下标计算地址：

```text
K[batch, position, head, dim]
V[batch, position, head, dim]
```

Paged KV layout 需要经过 block table：

```text
logical token position
  -> logical block index
  -> physical block id
  -> physical block base address
  -> offset inside block
```

这会引入：

- 非连续访存；
- block table lookup；
- boundary handling；
- warp coalescing 变差；
- block size 与性能之间的 tradeoff；
- variable-length sequence 下的越界处理。

### 3.2 Variable-length batch 更接近真实 serving

真实请求长度不同：

```text
seq_lens = [128, 512, 4096, 8192, 32768, ...]
```

kernel 需要支持每个 batch item 不同长度，不能只处理固定 shape 的 toy case。

这要求实现时显式处理：

- 每个 sequence 的实际 context length；
- 每个 sequence 对应的 block table；
- 最后一个 block 未填满的情况；
- 不同 sequence 长度导致的 load imbalance。

### 3.3 长上下文需要 streaming / online softmax

长 context 下不能简单把所有 attention score 全部存下来再 softmax。更现实的做法是 block-wise streaming softmax：

- 每个 tile 计算局部 score；
- 维护 running max；
- 维护 running sum；
- 逐 block 更新输出向量；
- 保证 FP16/BF16 下数值稳定。

这里的难点不只是“写出公式”，而是：

- 正确性；
- 数值稳定；
- 内存占用；
- bandwidth 利用率；
- 实现复杂度。

### 3.4 Benchmark 容易误导

如果只测一个 batch、一个 context length、一个 GPU，然后给出“快了 x 倍”，说服力很弱。

本项目需要系统化 benchmark：

- batch size：1 / 2 / 4 / 8 / 16 / 32
- context length：128 / 512 / 2K / 4K / 8K / 16K / 32K
- head dim：64 / 128
- block size：8 / 16 / 32 / 64
- dtype：FP16 / BF16
- metrics：latency、tokens/s、effective bandwidth、显存占用、p50 / p95
- baselines：PyTorch dense SDPA、PyTorch paged reference、FlashInfer、Triton implementation（CUDA extension 门开后加入）

真正有价值的是解释性能边界，而不是只追一个漂亮数字。

## 4. 三层交付结构

暑假版本按“Triton 单主线 + 门控增强”推进：Level 1 拆成 Must / Should / Optional，Level 2 是 Week 4 末门控后的三选一，Level 3 是可选加分项。CUDA extension 不是并行任务线，而是门控选项——两个讲不透的 kernel 不如一个讲得透的。

### Level 1: 主线（Must / Should / Optional）

Level 1 的目标是形成一个完整、可信、可复现的 Paged-KV Attention Triton kernel 项目。Must 全部完成即达到第一个可投递检查点（Week 4 末）。

Must（缺一不可）：

- PyTorch dense reference（FP32 ground truth）；
- PyTorch paged reference；
- block-table generator：随机乱序、碎片化、未用 slot 填垃圾值；
- Triton kernel：`q_len = 1`、variable-length batch、block table、`head_dim = 128`、FP16；
- correctness tests（覆盖清单见第 9 节）；
- benchmark：CSV、主配置 sweep、warmup/repeat、p50/p95、CUDA events 计时、时钟与环境记录；
- baselines：PyTorch dense SDPA、PyTorch paged reference、FlashInfer（见 10.3）；
- 带宽利用率指标（见 10.1）；
- profiling report 初稿：memory-bound 论证、paged 间接寻址代价、online softmax 开销、小 batch 长 context 的占用问题。

Should（时间允许则做，Week 6 前补齐）：

- `head_dim = 64`、BF16；
- block size sensitivity sweep（8 / 16 / 32 / 64）；
- 32K 长 context 覆盖；
- NCU 深度分析（若 Week 0 验证权限可用）；
- latency vs context length、tokens/s vs batch size、block size sensitivity 全套图表。

Optional：

- 完整 shape grid；
- roofline 图。

### Level 2: Week 4 末门控三选一

CUDA 门控开门条件（三条全部满足）：

1. Level 1 Must 全绿；
2. 检查点 1 文档完成；
3. lab notes 显示 kernel / profiling 阶段享受多于煎熬。

无论是否开门，都交付 `docs/cuda-design-sketch.md`：线程块到 (batch, head) 的映射、K/V block 的 shared memory staging、online softmax 的 warp shuffle 归约、向量化加载、与 Triton 自动处理部分的对照。它直接预答“用 CUDA 你会怎么写”这类面试问题；若开门，它就是实现计划。

#### 默认: split-KV（Flash-Decoding 风格）

`q_len = 1` 时每个 (batch, head) 只有一个线程块顺序扫 KV，小 batch 长 context 下 SM 大面积闲置——这是 Week 4 benchmark 会亲手暴露的问题。split-KV 把 context 分段并行计算 partial softmax，再用 reduce kernel 合并。先测出问题、再修掉它的前后对比图，是整个项目面试价值最高的一页，且与主线同一套 Triton 技术栈。

#### 门开且投算子岗: 最小 CUDA/C++ PyTorch Extension

范围锁死 `head_dim = 128` / MHA / FP16。重点展示：

- PyTorch custom op 集成；
- C++ binding；
- CUDA kernel launch；
- 复用既有 correctness tests 与 reference 对齐；
- 与 Triton 版本的接口和性能对照。

本质是移植一个已被测试覆盖、自己深刻理解的算法，不是从零设计。不要求超过 Triton，也不要求覆盖全部 shape grid。

#### 兴趣转向 infra: Mini KV Block Allocator + Request Simulation

如果 lab notes 显示你对 block 生命周期、调度和 benchmark 设计比 kernel 内部更兴奋，选这条。重点展示分页策略如何影响：

- fragmentation；
- block reuse；
- 显存占用；
- request arrival / finish；
- serving trace；
- continuous batching 下的 KV block 管理。

它能说明 paged KV cache 为什么是 serving 系统问题，而不只是 kernel layout 问题。

### Level 3: 可选加分

Level 3 只在 Level 1 完整、Level 2 已经做出深度后再考虑。

优先级：

1. GQA / MQA 支持（Triton 里近乎一行索引映射，成本低，Week 6 顺手做）；
2. FlashInfer / vLLM 设计层对照深化（FlashInfer 作为 benchmark baseline 已提前进 Level 1，这里只做设计与接口分析）；
3. INT8 KV cache。

GQA / MQA 优先级最高。原因是现代 LLM 场景里 GQA / MQA 很常见，而且实现复杂度比 quantized KV 更可控。

FlashInfer / vLLM 对照只做设计和接口层 comparison，不承诺性能追赶成熟库。

INT8 KV cache 是加分项，但不建议作为暑假主线。它会额外引入量化误差、scale layout、dequant 开销和 benchmark 解释成本。

## 5. 项目边界

### 5.1 要做

- 实现 decode attention，不做完整 transformer。
- 支持 `q_len = 1` 的 LLM generation 场景。
- 支持 paged KV cache layout。
- 支持 variable-length batch。
- Level 1 必须完成 PyTorch reference / Triton kernel / tests / benchmark / profiling。
- Level 2 按 Week 4 末门控三选一：split-KV（默认）/ 最小 CUDA extension / mini allocator。

### 5.2 不做

- 不复刻完整 vLLM。
- 不做完整 HTTP serving engine。
- 不做 tokenizer / model loading / sampling 全链路。
- 不承诺超过 vLLM、FlashInfer、TensorRT-LLM 等成熟库。
- 不在第一阶段做分布式推理。
- 不把 INT8 KV cache 作为主线交付。

合理目标是：

> 做一个教学级但工程严肃的 paged KV decode attention kernel，实现正确、可测、可解释，并在 naive PyTorch paged reference 之上取得明确性能优势。

## 6. 核心数据结构

### 6.1 输入张量

建议最小接口：

```python
def paged_attention(
    q,              # [batch, num_heads, head_dim]
    k_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables,   # [batch, max_num_blocks_per_seq]
    seq_lens,       # [batch]
    scale: float,
) -> torch.Tensor:
    # output: [batch, num_heads, head_dim]
```

### 6.2 Block table

对于 batch 中第 `b` 个请求：

```text
block_tables[b] = [physical_block_id_0, physical_block_id_1, ...]
seq_lens[b] = actual_context_length
```

逻辑位置到物理位置：

```text
logical_block_id = token_pos // block_size
offset           = token_pos % block_size
physical_block   = block_tables[b, logical_block_id]
```

### 6.3 MHA / MQA / GQA

第一阶段建议只支持 MHA：

```text
num_heads == num_kv_heads
```

第二阶段再支持 GQA / MQA：

```text
kv_head_id = query_head_id // group_size
```

## 7. 六周暑假路线

详细执行计划、每周任务与门控条件见 `ROADMAP.md`，此处只保留总览。

| 周期 | 目标 | 检查点 |
| --- | --- | --- |
| Week 0（2-3 天） | AutoDL 环境验证（重点：NCU 权限）、git/CI 工程化、必读材料 | — |
| Week 1 | dense/paged reference + block-table generator + correctness tests | — |
| Week 2-3 | Triton v0 → v1（先在连续 layout 上写对 online softmax，再加 block table 与 variable-length） | — |
| Week 4 | benchmark grid + FlashInfer/SDPA baseline + 带宽利用率 + profiling | ✅ 检查点 1：可投递 |
| Week 4 末 | CUDA 门控判定 + `docs/cuda-design-sketch.md` | — |
| Week 5 | 门控三选一：split-KV（默认）/ 最小 CUDA port / mini allocator | ✅ 检查点 2 |
| Week 6 | GQA + 最终报告 + limitations（含 vAttention）+ 简历定稿 + 面试自测 | ✅ 最终形态 |

两条节奏纪律：

- README 与简历 snippet 每周随进度更新，保证任何一周被打断，项目都是完整的小故事；
- 每周写约 300 字 lab note（最难的 bug、学到什么、享受/排斥什么，区分“工具链痛苦”与“方向排斥”），作为暑期结束时判断方向适配度的原始数据。

## 8. 推荐 repo 结构

```text
paged-kv-attention-kernel-lab/
  README.md
  pyproject.toml
  setup.py
  src/
    paged_kv_attention/
      __init__.py
      reference.py
      triton_kernels.py
      cuda_extension.py
      layouts.py
      benchmark_utils.py
  csrc/
    paged_attention.cpp
    paged_attention_kernel.cu
  tests/
    test_reference.py
    test_triton.py
    test_cuda_extension.py
    test_random_block_tables.py
  benchmarks/
    bench_decode_attention.py
    bench_block_size.py
    bench_context_length.py
  docs/
    design.md
    env-notes.md
    reading-list.md
    profiling-report.md
    benchmark-results.md
    cuda-design-sketch.md
    lab-notes/
  scripts/
    run_tests.sh
    run_benchmarks.sh
```

## 9. Correctness 测试标准

必须覆盖：

- batch size = 1 / 多 batch；
- context length 小于一个 block；
- context length 刚好等于 block size；
- context length 跨多个 block；
- 最后一个 block 未填满；
- block table 非连续；
- block table 随机顺序；
- FP32 reference；
- FP16 / BF16 tolerance；
- head_dim = 64 / 128；
- MHA first，GQA optional。

建议 tolerance：

```text
FP32: atol=1e-5, rtol=1e-5
FP16: atol=1e-2, rtol=1e-2
BF16: atol=2e-2, rtol=2e-2
```

实际 tolerance 需要根据实现和硬件结果微调，并在 README 中说明。

## 10. Benchmark 设计

### 10.1 Metrics

- latency per decode step（CUDA events 计时，warmup / repeat）
- tokens/s
- effective bandwidth utilization：`seq_len × 2 × num_kv_heads × head_dim × dtype_size / latency`，除以硬件峰值带宽得到百分比——既是 memory-bound 的硬证据，也是 speedup 的理论天花板
- peak memory allocated
- p50 / p95 latency（测量前用 `nvidia-smi -lgc` 锁时钟，或至少记录时钟状态）
- speedup over PyTorch paged reference（附条件说明）

### 10.2 Shape grid

```text
batch_size:     1, 2, 4, 8, 16, 32
context_length: 128, 512, 2048, 4096, 8192, 16384, 32768
num_heads:      8, 16, 32
head_dim:       64, 128
block_size:     8, 16, 32, 64
dtype:          fp16, bf16
```

不一定一开始全跑。先选一个主配置：

```text
batch_size = 8
num_heads = 16
head_dim = 128
block_size = 16
dtype = fp16
context_length = 128..16384
```

### 10.3 Baselines

常驻（Level 1 即包含）：

- PyTorch dense SDPA（连续 layout 参照，回答“paging 付出了什么代价”）
- PyTorch paged reference（教学下界）
- FlashInfer `BatchDecodeWithPagedKVCacheWrapper`（生产水位参照，只测不追）
- Triton paged kernel（本项目）

可选：

- CUDA extension paged kernel（门开后）
- vLLM paged attention 设计对照（Level 3，只做设计层）

## 11. Profiling 报告应回答的问题

报告不要只截图。至少回答：

1. 这个 kernel 是 compute-bound 还是 memory-bound？
2. 随 context length 增长，latency 为什么这样变化？
3. block size 对性能有什么影响？
4. Paged layout 相比连续 layout 付出了什么代价？
5. online softmax 带来的额外开销在哪里？
6. 小 batch + 长 context 下 SM 占用率为什么低？split-KV 能带来什么？
7. 如果做了 CUDA port，Triton 和 CUDA 版本差异在哪里？如果做了 allocator，分页策略如何影响 fragmentation 和 block reuse？
8. 当前实现与 FlashInfer 的实测差距是多少？可能来自哪里？

## 12. 最终交付物

必须交付：

- 可运行代码；
- 测试；
- benchmark；
- 图表；
- profiling report；
- README；
- 一段可放简历的项目描述。

建议最终文档：

```text
README.md
docs/design.md
docs/env-notes.md
docs/reading-list.md
docs/benchmark-results.md
docs/profiling-report.md
docs/cuda-design-sketch.md
docs/lab-notes/
```

## 13. 简历表述草案

简历 snippet 按实际交付状态分版本维护在 `RESUME_SNIPPETS.md`：

- 检查点 1 版本：Week 4 末 Level 1 完成即可投递；
- 版本 A：Triton 主线 + split-KV（无 CUDA extension）；
- 版本 B：门开后含最小 CUDA extension；
- 版本 C：Week 5 选 allocator。

纪律：只声称实际交付的内容；性能数字绑定具体硬件与配置；不写“复刻 vLLM”。

## 14. 风险与控制

### 风险 1: 一开始就想追成熟库性能

控制：

- 第一目标是正确性和可解释性；
- 只要求明显优于 naive PyTorch reference；
- 不承诺超过成熟库。

### 风险 2: Level 2 贪多

控制：

- CUDA extension 只在 Week 4 末门控通过后开做，且范围锁死；
- 默认 Level 2 是 split-KV，与主线同一套 Triton 技术栈；
- 不在 Week 5 同时开两条线；
- 无论门开与否，`docs/cuda-design-sketch.md` 兜底 CUDA 面试问题。

### 风险 3: Benchmark 没有说服力

控制：

- 提前定义 shape grid；
- 记录 warmup / repeat；
- 输出 p50 / p95；
- 保留原始 CSV；
- 图表从 CSV 自动生成。

### 风险 4: 项目过大

控制：

- 不做完整 serving；
- 不做 model integration；
- 不做多 GPU；
- 不把 quantized KV 作为主线；
- 主线只围绕 `paged_attention(q, k_cache, v_cache, block_tables, seq_lens)`。

### 风险 5: AutoDL profiling 权限不可用

表现：共享容器常见 `ERR_NVGPUCTRPERM`，Nsight Compute 读不到 performance counters。

控制：

- Week 0 花几块钱先验证；
- fallback：CUDA events + `torch.profiler` + 解析法带宽模型（理论读取量 / 实测 latency，对比峰值带宽）；如果 `nsys` 可用，再补 timeline；
- profiling report 的论证不硬依赖 NCU。

### 风险 6: AI 生成代码掏空学习价值

表现：项目完成了，但面试被问“online softmax 的 rescale 为什么数学上成立”时答不出。

控制：

- Triton kernel 核心循环第一稿自己写，AI 只做 review 和 debug 陪练；
- 脚手架、测试、画图放心交给 AI；
- 每周合卷复述 online softmax 更新式与 block table 地址计算；
- lab notes 记录亲手 debug 的 bug，作为 ownership 证据。

## 15. 建议技术栈

- Python 3.10+
- PyTorch
- Triton
- CUDA Toolkit
- pytest
- pandas / matplotlib
- Nsight Systems
- Nsight Compute
- FlashInfer（benchmark baseline + 接口对照）

## 16. 判断标准

暑假结束时，如果能回答下面这些问题，就说明项目达到了简历级质量：

1. 为什么 decode attention 容易 memory-bound？
2. KV cache 为什么需要分页？
3. block table 如何把 logical token position 映射到 physical block？
4. Paged layout 对 coalescing 有什么影响？
5. online softmax 怎么保证数值稳定？
6. 你的 Triton kernel 如何组织 block-wise / online softmax？
7. benchmark 覆盖了哪些 shape？为什么这些 shape 有代表性？
8. profiler 证明了什么瓶颈？
9. 当前实现和 vLLM / FlashInfer 这类成熟库相比差在哪里？
10. 如果继续优化，下一步会做什么？

## 17. 一句话总结

这个项目的价值不在于“又做了一个 LLM 项目”，而在于它把一个真实 inference infra 痛点下钻到 kernel layout、custom op、benchmark 和 profiler 证据链。

如果交付完整，它可以把简历中的 CUDA 表述从“能读懂并修改现有 kernel”推进到：

> 独立实现并评估 LLM 推理核心自定义算子。
