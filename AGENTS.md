# 项目 Agent 协作规则

本仓库是面向 LLM decode inference（LLM 解码阶段推理）的 Paged-KV attention kernel lab（分页 KV 注意力内核实验室）。协作目标不是“尽快替用户写完”，而是帮助用户形成可解释、可复现、可面试讲清楚的工程项目。

## 交流与教学节奏

- 默认使用中文交流。
- 英文专业名词使用 `English（中文翻译）` 表达，例如 `Triton kernel（Triton 内核）`、`online softmax（在线 softmax）`、`block table（块表）`。
- 每个重要 step（步骤）开始前，先讲清楚：
  - 这个 step 要解决什么问题；
  - 输入、输出和正确性标准是什么；
  - 推荐的实现顺序是什么；
  - 容易出错的边界条件是什么。
- 用户优先自己写核心实现；用户写完后，agent 负责 review（审查）、debug（调试）、整理和规范化。
- 当用户产出比较凌乱时，不直接否定；先识别可保留的核心思路，再重构为项目风格一致的版本。

## 每个 Step（步骤）的协作模板

- 先说明本 step 在 roadmap（路线图）里的位置，以及它服务哪个 acceptance criterion（验收标准）。
- 再定义接口：输入 tensor（张量）、输出 tensor、shape（形状）、dtype（数据类型）、layout（布局）和 device（设备）。
- 接着列出正确性检查：需要和谁对齐、使用什么 tolerance（容差）、覆盖哪些边界 case（用例）。
- 然后给用户一个可完成的小任务，而不是一次性要求完成整个模块。
- 用户写完后，先检查逻辑，再检查代码风格，最后补测试和文档。
- 每个 step 结束时，明确 next step（下一步）和 remaining risk（剩余风险）。

## 学习边界

- `Triton paged attention kernel（Triton 分页注意力内核）` 的核心循环应先由用户尝试书写。
- Agent 可以提供草稿、伪代码、实现步骤、测试用例和调试建议，但要优先教会用户如何写，而不是直接替代用户完成核心学习部分。
- 对每个核心模块，推荐流程是：
  1. 解释概念和接口；
  2. 给出分步任务；
  3. 用户实现第一版；
  4. agent review/debug；
  5. agent 将凌乱产出规范化；
  6. 补测试和文档。
- 对非核心学习模块，例如 test harness（测试框架）、benchmark harness（基准测试框架）、plotting（画图）、CI（持续集成）和文档脚手架，agent 可以更主动实现。

## 用户初稿规范化规则

- 先保留能证明 ownership（所有权）的关键代码路径和推导痕迹。
- 再统一命名、shape check（形状检查）、error message（错误信息）、docstring（文档字符串）和测试组织。
- 如果用户的代码能工作但结构松散，优先整理为清晰的小函数和可测试接口。
- 如果用户的代码有 correctness bug（正确性缺陷），先写或指出能复现 bug 的测试，再修实现。
- 规范化后要说明哪些部分保留了用户原始思路，哪些部分做了工程化整理。

## 项目优先级

- 先完成 Triton correctness、benchmark、profiling 与 split-KV，再串行开始 CUDA/C++ extension（CUDA/C++ 扩展）。
- CUDA 主线范围锁死为 single-pass paged decode port；CUDA split-KV 仅作为 stretch goal，不与 Triton split-KV 并行开发。
- 遵守 `ROADMAP.md` 的交付顺序和 `ACCEPTANCE_CRITERIA.md` 的验收标准。
- 每周都要让项目处于可讲述状态：README、lab notes（实验笔记）、scripts（脚本）和 resume snippets（简历片段）随进度更新。
- 优先做一个讲得透、测得准的 kernel（内核）和 benchmark story（基准测试叙事），不要堆多个浅层实现。

## 代码质量与改动尺度

- 不默认追求最小改动；为了更好的正确性、可维护性和整体风格，可以接受较大范围修改。
- 扩大改动范围前，要说明原因、影响文件和验证方式。
- 保持项目风格一致：命名、目录、测试组织、docstring（文档字符串）、类型假设和脚本入口都要统一。
- 避免临时补丁掩盖根因；优先修正接口语义、layout（布局）定义、mask（掩码）逻辑和数值稳定性问题。
- 不做无关重构；每次整理都应服务于当前功能分节。

## 命名与语义优先

- 新增或重命名文件、测试、函数和文档标题时，优先表达语义角色，而不是学习周次或临时脚手架状态。
- 推荐使用 `reference_contracts`、`reference_testing`、`triton_decode_v0` 这类描述职责的名称；避免把新产物命名为 `week1_scaffold`、`week2_tmp` 之类时间或临时状态。
- 只有 roadmap（路线图）、lab notes（实验笔记）和明确按时间组织的计划文档才保留 `Week N` 表达。
- 文档引用应跟随语义化名称同步更新；如果旧文件名只是历史真实路径，可以保留引用，但要优先考虑重命名为更清楚的语义名称。
- 测试文件名应说明测试职责，例如 contract tests（契约测试）、semantic tests（语义测试）、alignment tests（对齐测试）或 benchmark tests（基准测试）。

## 实现规则

- Python package（Python 包）代码放在 `src/paged_kv_attention/`。
- Tests（测试）放在 `tests/`；GPU-only tests（仅 GPU 测试）必须使用 `gpu` pytest marker（pytest 标记）。
- Scripts（脚本）放在 `scripts/`，并能从 repo root（仓库根目录）复现运行。
- Python 版本和格式遵守 `pyproject.toml`：Python 3.10+，Ruff line length（行宽）100。
- correctness（正确性）默认以 FP32 dense reference（FP32 稠密参考实现）为 ground truth（真值）。
- Tensor layout（张量布局）、block-table semantics（块表语义）、dtype（数据类型）和 shape restriction（形状限制）必须在函数、docstring 或测试中明确。

## 下载与环境变更

- 用户要求下载依赖、工具链或其他大型产物时，正式下载前先做小样本测速；Python package source（包源）优先运行 `python scripts/benchmark_download_sources.py`，其他下载使用等价的限量流式探测。
- 测速至少区分 index/metadata latency（索引/元数据延迟）与实际文件吞吐，不能只用 ping 或小网页响应代替下载速度。
- 根据实测吞吐、待下载体积和当前磁盘余量，先估算下载时间与落盘开销，再选择候选源和安装方案。
- 如果只有唯一可信源，仍先测速并向用户说明预计耗时；速度异常时先排查代理、镜像与源配置，不直接开始长时间下载。
- 测速是瞬时网络事实，只用于当前下载决策；除非会影响环境复现，不把单次速度写成长期固定结论。

## Correctness（正确性）要求

- 覆盖 batch=1 和 multi-batch（多 batch）。
- 覆盖 context length（上下文长度）小于一个 block（块）、正好一个 block、跨多个 block、最后一个 block 未填满。
- 覆盖 non-contiguous（非连续）和 random-order（随机顺序）block table。
- 未用 slot（槽位）要填 garbage values（垃圾值），用于捕捉 out-of-bounds read（越界读取）。
- FP16/BF16 tolerance（容差）要相对 FP32 dense reference 验证。
- 不为了扩大 benchmark（基准测试）而牺牲 correctness coverage（正确性覆盖）。

## Benchmark（基准测试）与 Profiling（性能剖析）

- GPU kernel latency（GPU 内核延迟）测量使用 CUDA events（CUDA 事件）。
- benchmark 输出必须包含 warmup（预热）、repeat（重复次数）、p50、p95。
- 结果中记录环境事实：GPU 型号、driver（驱动）、CUDA、PyTorch、Triton、clock state（时钟状态）和关键包版本。
- 先输出 CSV，再生成图表。
- 使用 analytical bytes read（解析法读取字节数）/ measured latency（实测延迟）计算 effective bandwidth（有效带宽），再与 hardware peak bandwidth（硬件峰值带宽）对比。
- baseline（基线）分开记录：PyTorch dense SDPA、PyTorch paged reference、FlashInfer、Triton implementation，以及 CUDA runtime checkpoint 后的 CUDA implementation。
- 如果 NCU 不可用，按文档 fallback（回退）到 `nsys`、`torch.profiler` 和 analytical bandwidth model（解析带宽模型）。

## 文档要求

- 环境事实写入 `docs/env-notes.md`，不要只留在聊天或终端历史里。
- 每周 lab note 放在 `docs/lab-notes/`，记录最难的 bug（缺陷）、从 profiler/source code（性能剖析器/源码）学到的内容、方向适配观察。
- Week 4 需要产出或更新 `docs/benchmark-results.md` 和 `docs/profiling-report.md`。
- CUDA 实现前必须产出 `docs/cuda-design-sketch.md`，覆盖 block mapping（块映射）、shared-memory staging（共享内存暂存）、online softmax reduction（在线 softmax 归约）、vectorized loads（向量化加载）和与 Triton 的对照。
- `RESUME_SNIPPETS.md` 必须和真实实现、真实测试、真实 benchmark 结果一致。

## 功能分节与提交节奏

- 以一组连贯功能为单位推进，例如：
  - dense/paged reference + layout tests；
  - block-table generator + garbage-slot tests；
  - Triton v0 kernel + correctness tests；
  - benchmark harness + CSV output；
  - profiling report + plots。
- 每个功能分节完成后再整理风格、补测试、更新文档，并准备 commit（提交）。
- commit 拆分以语义为主：功能实现、测试补充、文档整理、命名清理应在合理情况下分开提交，而不是按时间顺序或文件数量机械拆分。
- Git 历史同时承担学习路线记录。到达一个可复述、可验证的 learning checkpoint（学习整理节点）时，再集中整理工作树和准备提交。
- checkpoint 优先使用以下类型：`concept`（概念和推导成立）、`kernel`（核心实现首次跑通）、`correctness`（边界测试与 reference 对齐）、`performance`（benchmark/profiling 得出可解释结论）。
- 一个 checkpoint 可以包含 1-3 个语义 commit，例如实现、测试和笔记；这些 commit 应共同讲清楚该节点解决了什么、如何验证、还剩什么限制。
- commit subject（提交标题）描述真实技术内容，例如 `Implement continuous dense Triton decode attention`，不使用 `week2 progress`、`v0 done` 等时间或临时阶段名称。
- 不把半成品核心逻辑伪装成完成状态；如果功能只完成一部分，要明确标注 remaining work（剩余工作）。
- 不自动 commit；用户确认后再执行提交。

## 验证命令

优先运行最相关的命令：

```bash
bash scripts/run_tests.sh
```

GPU smoke test（GPU 冒烟测试）：

```bash
python scripts/gpu_smoke.py
```

新增 GPU tests 时，要确保 CPU-only CI（仅 CPU 持续集成）仍然能在没有 CUDA device（CUDA 设备）的环境中运行。
