<p align="center">
  <a href="https://github.com/wangkingh/FastXC_ultra/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/wangkingh/FastXC_ultra/ci.yml?branch=main&label=CI&logo=github" alt="CI Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/wangkingh/FastXC_ultra?color=blue&logo=open-source-initiative" alt="MIT License">
  </a>
  <a href="https://github.com/wangkingh/FastXC_ultra/stargazers">
    <img src="https://img.shields.io/github/stars/wangkingh/FastXC_ultra?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/wangkingh/FastXC_ultra/issues">
    <img src="https://img.shields.io/github/issues/wangkingh/FastXC_ultra?logo=github" alt="Open issues">
  </a>
  <a href="https://github.com/wangkingh/FastXC_ultra/pulls">
    <img src="https://img.shields.io/github/issues-pr/wangkingh/FastXC_ultra?logo=github" alt="Open pull requests">
  </a>
  <img src="https://img.shields.io/github/last-commit/wangkingh/FastXC_ultra?logo=git" alt="last commit">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia" alt="CUDA >=11.8">
</p>
---

# FastXC Ultra
**High‑performance CPU‑GPU pipeline for single‑ and multi‑array ambient‑noise cross‑correlation (1‑ & 9‑component)**  

<!-- Switch language / 切换语言: [简体中文](README.zh-CN.md) -->
* Switch Language 切换语言: [English](README.md)[英文], [简体中文](README.zh-CN.md)[Simplified Chinese]

FastXC Ultra coordinates **CUDA‑C kernels** with a **Python controller** to turn raw SAC waveforms into stacked Noise‑Correlation Functions (NCFs) **in minutes, not hours**.

> **Heads‑up** &ndash; PWS/tf‑PWS stacking still needs **≥ 20 GB VRAM per GPU**.  
> Smaller cards can run linear stacking or shorter windows.

---

## 🚩 Key features
| Category | Highlights |
|----------|------------|
| **Speed** | CUDA‑accelerated kernels for `sac2spec_ultra`, `xc_fast`, `ncf_pws` |
| **Flexibility** | Single‑array **or** dual‑array XC, single‑component **or** 3×3 = 9‑component workflows |
| **Stacking** | Linear, **PWS**, **tf‑PWS** with GPU acceleration |
| **Automation** | One INI config + `FastXCPipeline` with per‑step modes (`SKIP`, `CMD_ONLY`, …) |
| **Clean I/O** | Regex‑based SAC search, auto file‑lists, incremental NCF writing |

---

## 🌱 Quick install
```bash
# 0. Clone
git clone https://github.com/wangkingh/FastXC_ultra
cd FastXC_ultra

# 1. Build CUDA/C executables (required)
cd cuda_c_src && make veryclean && make    # edit ARCH in Makefile if needed

# 2. Check and modify the configuration file
vim *.ini and set the paramters

# 3. Check and modify the master script run.py
Check the path of the configuration file and set it appropriately

# 5. Run the full pipeline
python run.py
```

### Build tips
Change compute capability in `cuda_c_src/Makefile`, for example:
```make
export ARCH = sm_89      # RTX 4090 → CC 8.9
```
Verify with:
```bash
utils/check_gpu/compile.sh && utils/check_gpu/check_gpu
```

---

## 🏗 Directory snapshot
<details>
<summary>Click to expand</summary>

```text
cuda_c_src/          CUDA kernels + Makefiles
  ├─ sac2spec_ultra/   SAC → spectrum
  ├─ xc/               spectrum × spectrum
  ├─ ncf_pws/          PWS / tf‑PWS stacking
  ├─ rotate/           ZNE ↔ RTZ rotation
  └─ Makefile
fastxc/              Python orchestrator
  ├─ cmd_generator/
  ├─ cmd_deployer/
  ├─ list_generator/
  └─ utils/
bin/                 Pre‑built executables (post‑compile)
utils/               GPU checker, plots, helper scripts
config/              Example *.ini
run.py               Minimal launcher
```
</details>

---

## ⚙️ Pipeline & modes
`run.py` executes eight ordered steps:

| # | Step | Purpose |
|---|------|---------|
| 1 | GenerateFilter     | FIR/IIR design (`filter.txt`)          |
| 2 | OrganizeSAC        | Group SAC files by station/time        |
| 3 | Sac2Spec           | GPU FFT → `segspec/` spectra           |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                        |
| 5 | ConcatenateNcf     | Merge daily NCFs                       |
| 6 | Stack              | Linear / PWS / tf‑PWS                  |
| 7 | Rotate             | ENZ ↔ RTZ                              |
| 8 | Sac2Dat            | Optional SAC → DAT conversion          |

Each step supports the following modes:

| Mode | Action |
|------|--------|
| `SKIP`        | Do nothing |
| `PREPARE`     | Prepare directory/file‑list |
| `CMD_ONLY`    | Generate CLI commands |
| `DEPLOY_ONLY` | Execute commands (no generation) |
| `ALL`         | Perform everything above |

```python
from fastxc import StepMode, FastXCPipeline
pipe = FastXCPipeline("my.ini")
pipe.run({
    "CrossCorrelation": StepMode.CMD_ONLY,
    "Rotate": StepMode.SKIP,
})
```

---

## 📝 Config cheat‑sheet
Below is an annotated excerpt that matches the **2025‑05‑02** sample you provided.  
Generate a full, commented template with:
```bash
python -m fastxc --generate-template
```

```ini
[array1]                         ; ---- Data source #1 ----
sac_dir       = ./               # root folder
pattern       = {home}/{*}/{*}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
time_start    = 2017-09-01 00:00:00
time_end      = 2017-09-30 01:00:00
component_list= E,N,Z            # strict ENZ order!
time_list     = NONE
sta_list      = NONE

[array2]                         ; ---- Data source #2 (optional) ----
sac_dir       = NONE
pattern       = {home}/{YYYY}/{station}_{component}_{JJJ}.{suffix}
component_list= Z                # single‑component example
# other time_*, sta_list identical to array1

[preprocess]                     ; ---- Pre‑processing ----
win_len     = 3600               # seconds
shift_len   = 3600
delta       = 0.1                # sps = 10 Hz
normalize   = RUN-ABS-MF         # running‑abs & median filter
bands       = 0.1/0.5 0.5/1 1/2  # Hz, whitespace‑separated
whiten      = BEFORE             # pre‑XC whitening
skip_step   = -1                 # -1 → keep all windows, or 0,1,2,3

[xcorr]                          ; ---- Cross‑correlation ----
max_lag     = 100                # seconds
write_mode  = APPEND             # incremental NCF
write_segment = False            # write per segment in each winlen
distance_range = -1/50000        # km, limit XC to a range
azimuth_range  = -1/360          # azimuth, limit XC to a range
source_info_file = NONE          # for future usage    

[stack]                          ; ---- Stacking ----
stack_flag     = 100             # 1=linear 0=PWS 0=tf‑PWS → here linear only
sub_stack_size = 1
source_info_file = NONE          # for future usage

[executables]                    ; ---- Binaries ----
sac2spec = /path/to/bin/sac2spec_ultra
xc       = /path/to/bin/xc_multi_channel
stack    = /path/to/bin/ncf_pws
rotate   = /path/to/bin/RotateNCF

[device]                         ; ---- Hardware ----
gpu_list      = 0,1,2,3
gpu_task_num  = 1,1,1,1
gpu_mem_info  = 40,40,40,40     # GiB
cpu_count     = 100

[storage]                        ; ---- Output ----
output_dir = ./
overwrite   = True
clean_ncf   = True

[debug]                          ; ---- Debugging ----
dry_run        = False
log_file_path  = NONE
```

---

## 🖥 Environment check
```bash
nvidia-smi            # CUDA 11.8+ driver
nvcc --version        # confirm toolkit matches driver
python -m fastxc --doctor   # built‑in sanity checker
```

---

## ℹ️ FAQ
<details><summary>Can I run without Array‑2?</summary>

Yes. Set `[array2].sac_dir = NONE`. The pipeline will auto‑switch to single‑array mode.
</details>

<details><summary>How do I skip rotation?</summary>

Either omit the `[rotate]` section or force the step mode:
```python
pipe.run({"Rotate": StepMode.SKIP})
```
</details>

---

## 📒 Change Log
See [CHANGELOG.md](CHANGELOG.md) for details.

## 📧 Contact
Open an [issue](https://github.com/wangkingh/FastXC_ultra/issues) or reach the author:  
**Email:** <wkh16@mail.ustc.edu.cn>

---

© 2023‑2025 Wang Jingxi with ChatGPT O3 — Licensed under the **MIT** License.
