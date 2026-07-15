# Week 0 Prep: Environment + Repo Foundation

目标：用 2-3 天把项目从“计划”推进到“可执行实验室”。Week 0 不追求性能结果，重点是确认 GPU 工具链、固定环境组合、建立工程骨架，并完成最小阅读闭环。

## 0. 原则

- 先验证最贵/最容易卡住的部分：4090 GPU、Triton、CUDA extension 编译、NCU 权限。
- 本地只做低成本准备；GPU 机器只做必须在 GPU 上验证的事情。
- 所有环境事实写进 `docs/env-notes.md`，不要只留在聊天或终端历史里。
- 每份必读材料只读到能回答 Week 1-3 会用到的问题，单份 60-90 分钟 timebox。

## 1. Week 0 交付物

- `docs/env-notes.md`：GPU 型号、driver、CUDA、PyTorch、Triton、NCU 权限、fallback 结论。
- `docs/lab-notes/week0.md`：读书笔记、工具链验证过程、方向适配观察。
- Git repo 初始化完成，本地至少有一次 commit。
- GitHub repo 创建完成，远端 push 成功。
- CI 初稿：CPU 可跑 tests/lint；GPU tests 后续用 marker skip。
- GPU smoke script 跑通：`scripts/gpu_smoke.py`。

## 2. 本地准备

### 2.1 Repo 初始化

如果还没初始化：

```bash
git init
git add README.md ROADMAP.md ACCEPTANCE_CRITERIA.md docs scripts
git commit -m "Initialize paged-kv attention kernel lab plan"
```

建议本地目录保持当前命名：

```text
paged-kv-attention-kernel-lab
```

### 2.2 最小工程骨架

Week 0 只需要建结构，不需要实现 kernel：

```text
src/paged_kv_attention/
tests/
benchmarks/
scripts/
docs/lab-notes/
```

Week 1 再填 `reference.py`、`layouts.py` 和 tests。

### 2.3 GitHub / CI

CI 初稿目标：

- Python import smoke；
- CPU-only unit tests；
- lint / format check；
- GPU tests 用 `pytest.mark.gpu` 标记并默认 skip。

不要在 Week 0 为 CI 追求 GPU runner。GPU 验证先手动跑，结果写入 `docs/env-notes.md`。

## 3. AutoDL GPU 验证

### 3.1 租机建议

- 优先选 RTX 4090；
- 只租几小时；
- 选择带 PyTorch / CUDA 基础镜像的环境，减少安装时间；
- 开机后第一件事记录 `nvidia-smi` 和 Python package 版本。

### 3.2 需要跑通的三件事

1. PyTorch + Triton hello kernel；
2. CUDA extension 编译 smoke；
3. `ncu` 权限验证。

推荐先运行：

```bash
UV_HTTP_TIMEOUT=600 uv sync --locked --group dev
bash scripts/check_env.sh
uv run python scripts/gpu_smoke.py
```

再记录：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton import failed:", repr(exc))
PY
ncu --version || true
```

### 3.3 NCU 权限判定

对 smoke script 跑一次：

```bash
ncu --set full --target-processes all uv run python scripts/gpu_smoke.py
```

判定：

- 如果能输出 kernel 指标：Week 4 使用 NCU 做 profiling。
- 如果出现 `ERR_NVGPUCTRPERM`：记录错误原文，Week 4 fallback 到 `nsys + torch.profiler + 解析法带宽模型`。
- 如果 `ncu` 不存在：记录镜像缺失，先不在 Week 0 花太多时间修，除非安装成本很低。

当前容器只有容器内权限时，`ERR_NVGPUCTRPERM` 不是阻塞项。默认启用 fallback：CUDA events 测 latency（延迟）、`torch.profiler` 看 timeline（时间线）、analytical bandwidth model（解析带宽模型）估算 effective bandwidth（有效带宽）。

## 4. Week 0 阅读 Timebox

### 4.1 PagedAttention 论文

目标问题：

- 连续 KV cache 为什么浪费显存？
- block table 如何从 logical position 映射到 physical block？
- block size 太大/太小分别有什么代价？

输出：写 3-5 行到 `docs/lab-notes/week0.md`。

### 4.2 Hugging Face paged attention / continuous batching 文档

目标问题：

- `q_len=1` decode path 的语义是什么？
- `block_tables` 和 `cache_seqlens` 应该长什么形状？
- 本项目接口 `paged_attention(q, k_cache, v_cache, block_tables, seq_lens)` 每个参数为什么这样设计？

输出：写 3-5 行接口笔记。

### 4.3 Triton tutorials 01 / 02 / 03

目标问题：

- `program_id` 如何映射 row / block？
- `tl.load` / `tl.store` 的 mask 语义是什么？
- fused softmax 的 row-wise max / sum reduction 怎么写？
- matmul 教程里的 block loop 和 accumulator 模式如何迁移到 online softmax？

输出：不看教程能写出一个带 mask 的 row-wise softmax kernel。

## 5. Week 0 验收标准

Week 0 完成的最低标准：

- `scripts/gpu_smoke.py` 在 GPU 机器上跑通或留下明确失败原因。
- `docs/env-notes.md` 填完环境表和 NCU/fallback 判定。
- `docs/lab-notes/week0.md` 有三份必读材料的简短笔记。
- repo 初始化并能 push 到 GitHub。
- 下一步 Week 1 的首要任务明确：dense reference、paged reference、block-table generator、correctness tests。

## 6. 不做什么

- 不开始写正式 Triton paged attention kernel。
- 不追 FlashInfer baseline。
- 不做 benchmark grid。
- 不为 NCU 权限耗超过半天；失败就接受 fallback。
- 不把 CUDA extension 当主线，只验证编译工具链是否可用。

## 7. Week 1 启动条件

满足以下条件即可进入 Week 1：

- PyTorch + Triton 版本组合已记录；
- GPU smoke 结果已记录；
- NCU 可用性或 fallback 已记录；
- repo / docs / scripts 基本结构存在；
- 读书笔记足够支撑接口设计。
