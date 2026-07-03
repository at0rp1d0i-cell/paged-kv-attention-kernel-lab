# Roadmap

## 总原则

1. **Triton 单主线**：CUDA extension 不是并行任务线，而是 Week 4 末的门控选项（见下）。两个讲不透的 kernel 不如一个讲得透的。
2. **连续可投递**：README 和简历 snippet 每周随进度更新，保证任何一周被打断，项目都是一个完整的小故事。两个正式检查点：Week 4 末（Level 1 完成）与 Week 6 末（最终形态）。
3. **LLM 协作三规则**：
   - 测试、benchmark harness、画图、脚手架可以交给 AI；**Triton kernel 核心循环第一稿必须自己写**，AI 只做 reviewer 和 debug 陪练；
   - 每周合卷复述一次：手推 online softmax 的 running max / running sum 更新式和 block table 地址计算；
   - lab notes 记录亲手 debug 的 bug（mask 边界、tolerance、越界读），作为 ownership 证据。
4. **每周 lab note（约 300 字）**：本周最难的 bug、从 profiler / 源码学到什么、哪部分享受哪部分排斥；把“工具链痛苦”和“方向排斥”分开记——这是暑期结束时回答“我适不适合这个方向”的原始数据。

## 周计划总览

| 周期 | 目标 | 检查点 |
| --- | --- | --- |
| Week 0（2-3 天） | AutoDL 环境验证（重点：NCU 权限）、repo 工程化、必读材料 | — |
| Week 1 | dense/paged reference + block-table generator + correctness tests | — |
| Week 2-3 | Triton kernel v0 → v1，测试全绿，长 context 快过 paged reference | — |
| Week 4 | benchmark grid + FlashInfer/SDPA baseline + 带宽利用率 + profiling | ✅ 检查点 1：可投递 |
| Week 4 末 | CUDA 门控判定 + `docs/cuda-design-sketch.md` | — |
| Week 5 | 门控三选一：split-KV（默认）/ 最小 CUDA port / mini allocator | ✅ 检查点 2 |
| Week 6 | GQA + 最终报告 + limitations + 简历定稿 + 面试自测 | ✅ 最终形态 |

## Week 0: 环境验证 + 工程化（2-3 天）

- 租最便宜的 4090 几小时，完成三件事：
  - 确定并 pin 住 PyTorch + Triton 版本组合，跑通一个 hello Triton kernel；
  - 编译一个 vector-add CUDA extension（验证编译工具链，为 Week 5 可能的 CUDA port 留路）；
  - 对它跑一次 `ncu`，验证 GPU performance counter 权限。若报 `ERR_NVGPUCTRPERM`，确定 fallback：`nsys` + torch.profiler + 解析法带宽模型。
- `git init` + GitHub + CI（CPU 可跑的测试 + lint；GPU tests 打 `gpu` marker skip）。
- 必读（timebox 1.5 天，其余资料用到再查；完整分层阅读清单见 `docs/reading-list.md`）：
  - PagedAttention 论文（重点：设计章节与 block size 讨论）；
  - Hugging Face paged attention 文档（接口参考）；
  - Triton 官方教程 01 / 02 / 03（vector add、fused softmax、matmul）。
- 产出：`docs/env-notes.md`（GPU 型号、driver、版本组合、NCU 可用性、时钟设置）。

## Week 1: Reference + Layout + Tests

- 实现 dense decode attention reference（FP32 ground truth）。
- 实现 paged KV reference（定义 paged 语义）。
- block-table generator：随机乱序、碎片化分配、未用 slot 填垃圾值。
- correctness tests 全清单（见 ACCEPTANCE_CRITERIA）。
- 大部分工作在 Mac 本地完成，GPU 只做一次 smoke，省租金。

验收：dense/paged 对齐；variable-length batch 正确；随机 block table 测试通过。

## Week 2-3: Triton v0 → v1（核心爬坡段，两周）

- v0：batch=1、连续 layout（无 block table）、MHA、`head_dim=128`、FP16——目标是把 online softmax 在 Triton 里写对（fused softmax 教程的直接延伸）。
- v0.5：加 block table 间接寻址。
- v1：variable-length batch、最后 block 未满、`head_dim=64/128`、BF16。
- kernel 核心循环自己写；AI 只 review。

验收：correctness tests 全绿；至少一个长 context 配置快过 PyTorch paged reference。

若 Week 3 末 v1 未完成：压缩 Week 4 的 benchmark 范围，不砍测试。

## Week 4: Benchmark + Profiling（检查点 1）

- baselines：
  - PyTorch dense SDPA（回答“paging 付出了什么代价”）；
  - PyTorch paged reference（教学下界）；
  - FlashInfer `BatchDecodeWithPagedKVCacheWrapper`（生产水位参照，只测不追）。
- 主配置 sweep：context 128 → 32K；batch 1 → 32；block_size 16/32。
- 测量纪律：CUDA events、warmup/repeat、p50/p95、`nvidia-smi -lgc` 锁时钟（或记录时钟状态）、环境快照写入 CSV。
- 带宽利用率：`seq_len × 2 × num_kv_heads × head_dim × dtype_size / latency`，除以硬件峰值带宽得百分比——既是 memory-bound 的硬证据，也是 speedup 的理论天花板。
- profiling 按 Week 0 验证的工具链执行（NCU 或 nsys + 解析法）。
- 预期发现并解释：batch=1 长 context 下每个 (batch, head) 单线程块顺序扫 KV 导致 SM 占用不足（Week 5 split-KV 的铺垫）。

✅ 检查点 1：`docs/benchmark-results.md` 与 `docs/profiling-report.md` 初稿完成；README 可读可复现；简历 snippet 检查点版本可投递。

## Week 4 末: CUDA 门控判定

开门条件（三条全部满足）：

1. Level 1 Must 全绿；
2. 检查点 1 文档完成；
3. lab notes 显示 kernel / profiling 阶段享受多于煎熬。

无论是否开门，花 1-2 天写 `docs/cuda-design-sketch.md`：线程块到 (batch, head) 的映射、K/V block 的 shared memory staging、online softmax 的 warp shuffle 归约、向量化加载（float4）、与 Triton 自动处理部分的对照。这份文档直接预答“用 CUDA 你会怎么写”这类面试题；若开门，它就是实现计划。

## Week 5: 门控三选一（检查点 2）

- **默认：split-KV（Flash-Decoding 风格）**——context 分段并行计算 partial softmax，再用 reduce kernel 合并。验收：correctness 与 v1 对齐；batch=1/2 长 context 前后对比图；报告解释占用率变化。
- **门开且投算子岗：最小 CUDA port**——范围锁死 `head_dim=128` / MHA / FP16；复用既有测试对齐；与 Triton 接口和性能对照。移植已知正确、自己深刻理解的算法，不从零设计。
- **兴趣明确转向 infra：mini KV block allocator + request simulation**——arrival/finish、alloc/free、fragmentation、block reuse、memory trace。

## Week 6: GQA + 打包（最终形态）

- GQA / MQA 支持（Triton 里近乎一行索引映射）。
- 最终报告：limitations 必须包含 vAttention 反方证据（PagedAttention 不是唯一解，有间接寻址与 kernel 复杂度代价）。
- `scripts/run_tests.sh` / `scripts/run_benchmarks.sh` 复现脚本。
- 简历 snippet 按实际交付定稿（RESUME_SNIPPETS 版本 A / B / C）。
- 按 README 第 16 节十个问题做面试自测，答不出的补。
- 汇总 6 周 lab notes，写下方向判断结论：kernel、infra/系统、benchmark/performance，哪类工作让你愿意继续。
