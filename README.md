# FastXC 📈🌍 — GPU‑accelerated seismic cross‑correlation toolkit
[![Build](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/your-org/FastXC/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![CUDA ≥ 11.8](https://img.shields.io/badge/CUDA-11.8%2B-green?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)

FastXC combines **CUDA C kernels** with a **Python orchestration layer** to give you an
end‑to‑end, GPU‑powered workflow for:

1. **SAC → spectrum conversion**  
2. **Cross‑correlation (XC) & NCF production**  
3. **Phase‑weighted & TF‑PWS stacking**  
4. **Horizontal ↔ Radial rotation**  
5. **House‑keeping utilities** (segment extraction, SAC↔DAT, etc.)

All steps are configurable through a single `.ini` file and can be chained in one‑shot
with the `FastXCPipeline`.

> **TL;DR:** Install → prepare `config.ini` →  
> `python -m fastxc config.ini --steps='{"CrossCorrelation": "ALL"}'`

---

## ⏳ Quick start

```bash
# (1) clone & build CUDA executables
git clone https://github.com/your-org/FastXC.git
cd FastXC/cuda_c_src && make -j && cd ..

# (2) create a virtualenv
python -m venv .venv && source .venv/bin/activate
pip install -e .

# (3) copy & edit template config
python -m fastxc --generate-template   # produces template_config.ini.copy
vim template_config.ini.copy           # adjust paths / parameters

# (4) run the full pipeline
python -m fastxc template_config.ini.copy
```

---

## 🛠 Directory layout (hover to expand)

<details>
<summary><strong>Click to see</strong></summary>

```text
cuda_c_src/          # CUDA kernels (sac2spec, xc, rotate …) + Makefiles
fastxc/              # Python orchestrator
  ├── cmd_generator/ # create *.cmds.txt
  ├── cmd_deployer/  # dispatch cmds to GPUs / CPUs
  ├── list_generator/# build file‑lists fed to CUDA tools
  └── utils/         # helpers: config parser, design_filter, ...
utils/               # misc C helpers (check_gpu, extractSegments, …)
run.py               # minimal entry script example
```
</details>

---

## 📜 Configuration

Everything lives in one INI.  
The **bare minimum** you must tweak is:

```ini
[array_info1]
sac_dir   = /path/to/SAC
component_list = BHZ

[parameters]
output_dir = /path/to/out
delta      = 0.05
bands      = 0.1/0.5 0.5/1

[executables]
sac2spec = /abs/path/to/bin/sac2spec_ultra
xc       = /abs/path/to/bin/xc_fast
```

Need inspiration? Run:

```bash
python -m fastxc --generate-template
```

---

## 🚦 Pipeline steps & modes

<div style="overflow-x: auto">

| Step (ordered)          | Purpose                                 | Default mode | Typical tweaks |
|-------------------------|-----------------------------------------|-------------|----------------|
| `GenerateFilter`        | FIR/IIR design → `filter.txt`           | `ALL`       | Skip if you already have filters |
| `OrganizeSAC`           | Scan SAC tree, group by station/time    | `ALL`       | `PREPARE_ONLY` for dry runs |
| `Sac2Spec`              | FFT on GPU, produce `segspec/*.spec`    | `ALL`       | Often `SKIP` for pre‑computed spectra |
| `CrossCorrelation`      | GPU XC → `ncf/*.SAC`                    | `ALL`       | `CMD_ONLY` to inspect cmds |
| `ConcatenateNcf`        | Merge day‑wise NCFs                     | `ALL`       |                |
| `Stack`                 | linear / PWS / TF‑PWS stack             | `ALL`       | Turn off in `AGGREGATE` write‑mode |
| `Rotate`                | Z‑N‑E → R‑T or pairwise H‑R rotation    | `ALL`       | Needs 3‑comp data |
| `Sac2Dat`               | Convert SAC → DAT for AxiSEM, etc.      | `SKIP`      |                |

</div>

Use `StepMode` to fine‑grain execution:

```python
from fastxc import StepMode

steps = {
    "OrganizeSAC":   StepMode.PREPARE_ONLY,
    "CrossCorrelation": StepMode.CMD_ONLY,
    "Rotate":        StepMode.SKIP,
}
pipeline.run(steps)
```

---

## 🌊 ASCII seismogram (why not?)

```text
                         .  .                   .    .
                    .  .   .  .   .  .  .   .   .  .
_______________.___.___.___.___.___.___.___.___.___.__________________
```

---

## 💡 Pro tips

* **Scrollable tables / code**  
  Wrap them in `<div style="overflow-x:auto">…</div>` as shown above.
* **GPU sanity‑check**  
  `utils/check_gpu/check_gpu` prints compute‑capability & mem before the heavy jobs.
* **Dry‑run mode**  
  Set `dry_run = True` in `[advanced_debug]` to echo commands without executing.
* **Live log**  
  `log_file_path = fastxc_$(date +%F_%T).log` keeps every step’s output.

---

## 🖊 Citation

If you use *FastXC* in your research, please cite:

> Your Name et al. (2025). **FastXC: A GPU‑accelerated seismic cross‑correlation pipeline**. _Seismological Software Notes_, v1.0. https://doi.org/xx.xxxx/xxxx

---

## 📄 License

[MIT](LICENSE) © 2023‑2025 Your Name & Contributors
