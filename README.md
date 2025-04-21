<p align="center">
  <a href="https://github.com/your-org/FastXC/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/wangkingh/FastXC_ultra/ci.yml?branch=main&label=CI&logo=github" alt="CI Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/wangkingh/FastXC_ultra?color=blue&logo=open-source-initiative" alt="MIT License">
  </a>
  <a href="https://github.com/your-org/FastXC/stargazers">
    <img src="https://img.shields.io/github/stars/wangkingh/FastXC_ultra?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/your-org/FastXC/issues">
    <img src="https://img.shields.io/github/issues/wangkingh/FastXC_ultra?logo=github" alt="Open issues">
  </a>
  <a href="https://github.com/your-org/FastXC/pulls">
    <img src="https://img.shields.io/github/issues-pr/wangkingh/FastXC_ultra?logo=github" alt="Open pull requests">
  </a>
  <img src="https://img.shields.io/github/last-commit/wangkingh/FastXC_ultra?logo=git" alt="last commit">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia" alt="CUDA >=11.8">
</p>



* Switch language / 切换语言: [简体中文](README.zh-CN.md)

---

# FastXC
**High‑performance CPU‑GPU pipeline for ambient‑noise cross‑correlation (1‑ & 9‑component)**

FastXC orchestrates **CUDA‑C kernels** with a **Python controller** to turn raw SAC waveforms
into stacked Noise Correlation Functions — in minutes, not hours.

> **Heads‑up**: current *PWS* & *tf‑PWS* kernels require GPUs with **≥ 20 GB VRAM**.  
> Smaller cards can still run linear stacking or shorter windows.

---

## 🚩 Key features
| Category | Highlights |
|----------|------------|
| **Speed** | CUDA‑accelerated kernels for `sac2spec_ultra`, `xc_fast`, `RotateNCF` |
| **Flexibility** | Single‑array **or** dual‑array XC, single‑component **or** 3×3=9‑component workflows |
| **Stacking** | Linear, **PWS**, **tf‑PWS** with GPU acceleration |
| **Automation** | One‑file config (`.ini`) + `FastXCPipeline` with per‑step modes (`SKIP`, `CMD_ONLY`, …) |
| **Clean I/O** | Regex‑based SAC search, auto file‑lists, optional concatenation & DAT output |

---

## 🌱 Quick install
```bash
git clone https://github.com/your-org/FastXC
cd FastXC

# 1. build CUDA/C executables
cd cuda_c_src && make -j && cd ..

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -e .        # installs fastxc package + CLI

# 3. config
python -m fastxc --generate-template
vim template_config.ini.copy   # edit absolute paths

# 4. run full pipeline
python -m fastxc template_config.ini.copy
```

### Build tips
*Different GPU?* — edit `cuda_c_src/Makefile`  
```
export ARCH=sm_89   # e.g. RTX 4090 → CC 8.9
```
Check with:
```bash
utils/check_gpu/compile.sh && utils/check_gpu/check_gpu
```

---

## 🏗 Directory snapshot
<details>
<summary>Click to expand</summary>

```text
cuda_c_src/          CUDA kernels + Makefiles
fastxc/              Python orchestrator
  ├─ cmd_generator/  build *.cmds.txt
  ├─ cmd_deployer/   dispatch commands
  ├─ list_generator/ build file‑lists
  └─ utils/          config parser, filter design, …
config/              example *.ini
run.py               minimal launcher
```
</details>

---

## ⚙️ Pipeline & modes
FastXC runs eight ordered steps:

| # | Step | Purpose | CLI enum default |
|---|------|---------|------------------|
| 1 | GenerateFilter     | FIR/IIR design (`filter.txt`)          | `ALL` |
| 2 | OrganizeSAC        | Group SAC files by station/time        | `ALL` |
| 3 | Sac2Spec           | GPU FFT → `segspec/` spectra           | `ALL` |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                        | `ALL` |
| 5 | ConcatenateNcf     | Merge daily NCFs                       | `ALL` |
| 6 | Stack              | Linear / PWS / tf‑PWS                  | `ALL` |
| 7 | Rotate             | ZNE ↔ RT / HR                          | `ALL` |
| 8 | Sac2Dat            | Optional SAC → DAT conversion          | `SKIP` |

Use `StepMode` for overrides:

```python
from fastxc import StepMode, FastXCPipeline
pipe = FastXCPipeline("my.ini")
pipe.run({{
    "CrossCorrelation": StepMode.CMD_ONLY,
    "Rotate": StepMode.SKIP,
}})
```

---

## 📝 Config cheat‑sheet
```ini
[array_info1]
sac_dir = /data/array1
pattern = {{home}}/{{YYYY}}/{{station}}.{{component}}.{{suffix}}
component_list = E,N,Z
time_start = 2019-01-01 00:00:00
time_end   = 2019-12-31 23:59:59

[array_info2]
sac_dir = NONE
component_list = Z

[parameters]
output_dir = /data/out
delta      = 0.05
bands      = 0.1/0.5 0.5/1
max_lag    = 600
win_len    = 3600
shift_len  = 3600
stack_flag = 110

[executables]
sac2spec = ./bin/sac2spec_ultra
xc       = ./bin/xc_fast
stack    = ./bin/ncf_pws
rotate   = ./bin/RotateNCF

[device_info]
gpu_list    = 0,1
gpu_task_num= 1,1
gpu_mem_info= 40,40
cpu_count   = 32
```
Run `python -m fastxc --generate-template` for a full commented template.

---

## 🖥 Environment check
```bash
nvidia-smi
nvcc --version
```

---

## ℹ️ FAQ
<details><summary>Why no tf‑PWS in MULTI mode?</summary>
Early design assumed the GPU memory trade‑off; support is planned.
</details>

---

## 📜 References
* Wang et al. (2025) “High‑performance CPU‑GPU Heterogeneous Computing Method for 9‑Component Ambient Noise Cross‑Correlation.” *Earthquake Research Advances* (in press).  
* Bensen G.D. et al. (2007) *GJI* 169(3): 1239‑1260.  
* Cupillard P. et al. (2011) *GJI* 184(3): 1397‑1414.  
* Zhang Y. et al. (2018) *JGR* 123(9): 8016‑8031.

---

© 2023‑2025 Your Name & Contributors — Licensed under MIT
