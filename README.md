<p align="center">
  <a href="https://github.com/wangkingh/FastXC/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/wangkingh/FastXC_ultra/ci.yml?branch=main&label=CI&logo=github" alt="CI Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/wangkingh/FastXC_ultra?color=blue&logo=open-source-initiative" alt="MIT License">
  </a>
  <a href="https://github.com/wangkingh/FastXC/stargazers">
    <img src="https://img.shields.io/github/stars/wangkingh/FastXC_ultra?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/wangkingh/FastXC/issues">
    <img src="https://img.shields.io/github/issues/wangkingh/FastXC_ultra?logo=github" alt="Open issues">
  </a>
  <a href="https://github.com/wangkingh/FastXC/pulls">
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
git clone https://github.com/wangkingh/FastXC_ultra

cd utils

bash compile.sh

./chech_gpu

# 0. Check the output GPU info and rewrite cuda_c_src/Makefile's arch option

cd FastXC

# 1. build CUDA/C executables
cd cuda_c_src && make veryclean && make

# 2. Check Python env
pip install numpy pandas scipy obspy ...

# 3. generate template config or copy from github repository
python -m fastxc --generate-template
vim template_config.ini.copy   # edit absolute paths

# 4. change the config path in run.py and run full pipeline
python run.py
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
  ├─ sac2spec_stable/ converting sac to spectrum
  ├─ sac2spec_ultra/  converting sac to spectrum using some other filters
  ├─ xc/              calculating cross-spectrum between sepctrum
  ├─ ncf_pws/         stacking NCFs
  ├─ rotate/          Rotating NCFs from ENZ coordiate to RTZ coordinate
  ├─ Makefile
fastxc/              Python orchestrator
  ├─ cmd_generator/  build *.cmds.txt
  ├─ cmd_deployer/   dispatch commands
  ├─ list_generator/ build file‑lists
  └─ utils/          config parser, filter design, …
bin/
  ├─ sac2spec_stable
  ├─ sac2spec_ultra
  ├─ xc_fast
  ├─ ncf_pws
  ├─ RotateNCF
utils/
  ├─ check_gpu/    Check the GPUs' capability
  ├─ extract.py    Python scripts convert the "bigsac" to sac after Cross-Correlation
  ├─ GPU_vs_CPU.png

config/              example *.ini
run.py               minimal launcher
```
</details>

---

## ⚙️ Pipeline & modes
FastXC run.py runs eight ordered steps:

| # | Step | Purpose |
|---|------|---------|
| 1 | GenerateFilter     | FIR/IIR design (`filter.txt`)          |
| 2 | OrganizeSAC        | Group SAC files by station/time        |
| 3 | Sac2Spec           | GPU FFT → `segspec/` spectra           |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                        |
| 5 | ConcatenateNcf     | Merge daily NCFs                       |
| 6 | Stack              | Linear / PWS / tf‑PWS                  |
| 7 | Rotate             | ZNE ↔ RTZ                              |
| 8 | Sac2Dat            | Optional SAC → DAT conversion          |


Each step has an option to contrl the processin mode:
| # | Mode | Explain|
|---|------|-------------|
| 1 | SKIP      | Do Nothing                           |
| 2 |PREPARE    | Generate Directory or File List      |
| 3 |CMD_ONLY   | Generate Terminal Commands           |
| 4 |DEPLOY_ONLY| Deploy Commands (without Generattion)|
| 5 |ALL        | Do All Operations Above              |

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

## 📝 Config cheat‑sheet (For Further explaination, Check the template config file)
```ini
[array_info1]
sac_dir = /data/array1
pattern = {home}/{YYYY}/{station}.{component}.{JJJ}.{suffix}
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
<details><summary>What is the difference of this repository and 
  <a href="https://github.com/wangkingh/FastXC" target="_blank">FastXC</a>? ?
</summary>

1. **Higher-sample-rate support:**
    The CUDA batch implemention of *PWS* / *tf-PWS* has been redesigned, so the programe can handle data recorded at much higher sampling rate.

2. **Disk-Friendly NCF output:**
    In the **xc** stage, each segment’s NCF is now **appended to the end of the existing file** instead of being written to a brand‑new file.

3. **Removal of “dual” mode:**  
    We dropped the *dual* workflow. Cross‑correlation and stacking now run in a single high‑performance mode—faster, but it does consume more disk space.

4. **Time‑Stamp filtering:**  
    You can pass a time‑list file to include only those SAC files that fall on the Time-Stamp.

5. **Cleaner configuration:**  
    The config file has been overhauled: common settings are up front, while expert‑level parameters sit under some **advanced-** sections, making self‑configuration easier.

</details>

---

## 📒Change Log
See [Change Log](changelog.md)
## 📧Author Contact Information

If you have any questions or suggestions or want to contribute to the project, open an [issue](https://github.com/wangkingh/FastXC/issues) or submit a pull request.

For more direct inquiries, you can reach the author at:  
**Email:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

It will be my great pleasure if my code can provide any help for your research!

## 🙏Acknowledgements
We extend our sincere gratitude to our colleagues from the University of Science and Technology of China, the Institute of Geophysics, China Earthquake Administration, the Institute of Earthquake Forecasting, China Earthquake Administration, and the Institute of Geology and Geophysics, Chinese Academy of Sciences, for their __significant contributions__ during this program's testing and trial runs!

## 📜 References
Wang et al. (2025). ["High-performance CPU-GPU Heterogeneous Computing Method for 9-Component Ambient Noise Cross-correlation."](https://doi.org/10.1016/j.eqrea.2024.100357) Earthquake Research Advances. In Press.


Bensen, G. D., et al. (2007). ["Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements."](https://dx.doi.org/10.1111/j.1365-246x.2007.03374.x) Geophysical Journal International 169(3): 1239-1260.


Cupillard, P., et al. (2011). ["The one-bit noise correlation: a theory based on the concepts of coherent and incoherent noise."](https://doi.org/10.1111/j.1365-246X.2010.04923.x) Geophysical Journal International 184(3): 1397-1414.

---

© 2023‑2025 Wang Jingxi & ChatGPT O3 — Licensed under MIT
