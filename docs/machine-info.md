# Machine Info（机器信息）

Last updated（更新时间）: 2026-07-15 10:39:16 UTC

> This file is a machine snapshot（机器快照）. Re-run `bash scripts/collect_machine_info.sh` whenever the project moves to a different machine, GPU instance（GPU 实例）, or CUDA environment（CUDA 环境）.

## Update Checklist（更新清单）

- Re-run this snapshot after changing machine or GPU instance（GPU 实例）.
- Copy key GPU/CUDA facts into `docs/env-notes.md` when they affect benchmark（基准测试） or profiling（性能剖析） decisions.
- Keep benchmark CSV files tied to the machine snapshot used for that run.

## System（系统）

```text
hostname: autodl-container-a43440b6b9-dbab4138
user: root
uid/gid: uid=0(root) gid=0(root) groups=0(root)
working_dir: /root/paged-kv-attention-kernel-lab
kernel: Linux autodl-container-a43440b6b9-dbab4138 5.15.0-119-generic #129-Ubuntu SMP Fri Aug 2 19:25:20 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux

PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## CPU（处理器）

```text
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        52 bits physical, 57 bits virtual
Byte Order:                           Little Endian
CPU(s):                               128
On-line CPU(s) list:                  0-127
Vendor ID:                            GenuineIntel
Model name:                           Intel(R) Xeon(R) Gold 6459C
CPU family:                           6
Model:                                143
Thread(s) per core:                   2
Core(s) per socket:                   32
Socket(s):                            2
Stepping:                             8
Frequency boost:                      enabled
CPU max MHz:                          3001.0000
CPU min MHz:                          800.0000
BogoMIPS:                             6000.00
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                       VT-x
L1d cache:                            3 MiB (64 instances)
L1i cache:                            2 MiB (64 instances)
L2 cache:                             128 MiB (64 instances)
L3 cache:                             120 MiB (2 instances)
NUMA node(s):                         2
NUMA node0 CPU(s):                    0-31,64-95
NUMA node1 CPU(s):                    32-63,96-127
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
```

## Memory（内存）

```text
               total        used        free      shared  buff/cache   available
Mem:           754Gi        33Gi        77Gi       1.0Gi       643Gi       715Gi
Swap:             0B          0B          0B

MemTotal:       791192664 kB
MemFree:        81517628 kB
MemAvailable:   749831312 kB
SwapTotal:             0 kB
SwapFree:              0 kB
```

## Disk And Filesystem（磁盘和文件系统）

```text
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   18G   13G  60% /
overlay          30G   18G   13G  60% /

NAME      SIZE TYPE  MOUNTPOINT                                                        FSTYPE MODEL
loop0    49.3M loop
loop1    63.8M loop
loop3    91.7M loop
loop4    50.1M loop
loop5    63.8M loop
loop7      74M loop
loop8   115.1M loop
sda     894.3G disk                                                                           SAMSUNG MZ7L3960
├─sda1      1G part
└─sda2  893.2G part  /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so.580.76.05
nbd0        0B disk
nbd1        0B disk
nbd2        0B disk
nbd3        0B disk
nbd4        0B disk
nbd5        0B disk
nbd6        0B disk
nbd7        0B disk
nvme0n1     7T disk                                                                           HWE62P447T6L00LN
└─md0      14T raid5 /etc/hostname
nvme2n1     7T disk                                                                           HWE62P447T6L00LN
└─md0      14T raid5 /etc/hostname
nvme1n1     7T disk                                                                           HWE62P447T6L00LN
└─md0      14T raid5 /etc/hostname
nbd8        0B disk
nbd9        0B disk
nbd10       0B disk
nbd11       0B disk
nbd12       0B disk
nbd13       0B disk
nbd14       0B disk
nbd15       0B disk
nbd16       0B disk
nbd17       0B disk
nbd18       0B disk
nbd19       0B disk
nbd20       0B disk
nbd21       0B disk
nbd22       0B disk
nbd23       0B disk
nbd24       0B disk
nbd25       0B disk
nbd26       0B disk
nbd27       0B disk
nbd28       0B disk
nbd29       0B disk
nbd30       0B disk
nbd31       0B disk
```

## GPU And CUDA（GPU 和 CUDA）

```text
Wed Jul 15 18:39:17 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:BD:00.0 Off |                  N/A |
| 41%   31C    P8             20W /  575W |       0MiB /  32607MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

nvidia-smi query:
0, NVIDIA GeForce RTX 5090, 32607 MiB, 32110 MiB, 580.76.05, 00000000:BD:00.0, 575.00 W, 180 MHz, 405 MHz, 3090 MHz, 14001 MHz, 12.0

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0

NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2025 NVIDIA Corporation
Version 2025.1.1.0 (build 35528883) (public-release)

nsys: not found

lspci: not found
```

## Python And Packages（Python 和包）

```text
uv 0.11.26 (x86_64-unknown-linux-gnu)

Python 3.12.3
/root/paged-kv-attention-kernel-lab/.venv/bin/python
Python 3.12.3
/root/paged-kv-attention-kernel-lab/.venv/bin/python3

sys.executable /root/paged-kv-attention-kernel-lab/.venv/bin/python
sys.version 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]
platform Linux-5.15.0-119-generic-x86_64-with-glibc2.35
python_prefix /root/paged-kv-attention-kernel-lab/.venv
virtual_env /root/paged-kv-attention-kernel-lab/.venv
torch: 2.13.0+cu130
torch.cuda 13.0
torch.cuda.is_available True
torch.cuda.device_count 1
torch.cuda.device[0].name NVIDIA GeForce RTX 5090
torch.cuda.device[0].capability 12.0
torch.cuda.device[0].total_memory 33668988928
triton: 3.7.1
pytest: 9.1.1
ruff: 0.15.20
ninja: 1.13.0
flashinfer: 0.6.14
```

## Project Environment（项目环境）

```text
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "paged-kv-attention-kernel-lab"
version = "0.0.0"
description = "Paged-KV attention kernel lab for LLM decode inference experiments."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=2.0",
    "torch==2.13.0+cu130",
    "triton==3.7.1; platform_system == 'Linux'",
]

[dependency-groups]
dev = [
    "ninja>=1.11",
    "pytest>=8",
    "ruff>=0.5",
]
plot = [
    "matplotlib>=3.9",
]
baseline = [
    "cuda-toolkit[nvcc]==13.0.3",
    "flashinfer-python[cu13]==0.6.14",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires a CUDA-capable GPU",
]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py310"

[[tool.uv.index]]
name = "tuna-pypi"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

## Git And Repo State（Git 和仓库状态）

```text
/root/paged-kv-attention-kernel-lab
4336ed0
main

Tracked project files:
.github/workflows/ci.yml
.gitignore
ACCEPTANCE_CRITERIA.md
AGENTS.md
README.md
ROADMAP.md
benchmarks/results/decode_attention_flashinfer.csv
benchmarks/results/decode_attention_flashinfer_bandwidth_by_batch.png
benchmarks/results/decode_attention_flashinfer_batch_scaling.png
benchmarks/results/decode_attention_flashinfer_latency_p50_by_batch.png
benchmarks/results/decode_attention_program_saturation.csv
benchmarks/results/decode_attention_program_saturation_batch_scaling.png
benchmarks/results/pytorch_paged_reference_smoke.csv
docs/benchmark-results.md
docs/env-notes.md
docs/lab-notes/week0.md
docs/lab-notes/week1.md
docs/lab-notes/week2.md
docs/lab-notes/week4.md
docs/learning-syllabus.md
docs/machine-info.md
docs/profiling-report.md
docs/reading-list.md
docs/sync-workflow.md
docs/week0-prep.md
docs/week1-structure.md
note/benchmark-fundamentals.md
note/dense-decode-attention-implementation.md
note/online-softmax.md
note/pytorch-syntax-reference.md
note/reference-stage-attention-kv-cache.md
note/reference-stage-testing.md
note/triton-dense-decode.md
note/triton-paged-indexing.md
note/triton-syntax-reference.md
note/triton-vs-pytorch.md
pyproject.toml
scripts/check_env.sh
scripts/collect_machine_info.sh
scripts/flashinfer_smoke.py
scripts/gpu_smoke.py
scripts/plot_benchmarks.py
scripts/profile_decode_attention.py
scripts/run_benchmarks.py
scripts/run_benchmarks.sh
scripts/run_tests.sh
scripts/sync_from_remote.sh
scripts/sync_to_remote.sh
src/paged_kv_attention/__init__.py
src/paged_kv_attention/benchmark_utils.py
src/paged_kv_attention/block_table.py
src/paged_kv_attention/flashinfer_baseline.py
src/paged_kv_attention/layouts.py
src/paged_kv_attention/reference.py
src/paged_kv_attention/triton_decode.py
tests/test_benchmark_utils.py
tests/test_flashinfer_baseline.py
tests/test_import.py
tests/test_reference.py
tests/test_reference_contracts.py
tests/test_triton_decode.py
uv.lock
```

## Validation Commands（验证命令）

```bash
UV_HTTP_TIMEOUT=600 uv sync --locked --group dev
bash scripts/check_env.sh
bash scripts/run_tests.sh
uv run python scripts/gpu_smoke.py
uv run --group baseline python scripts/flashinfer_smoke.py
```
