<p align="center">
  <a href="https://github.com/your-org/FastXC/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/your-org/FastXC/ci.yml?branch=main&label=CI&logo=github" alt="CI Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/your-org/FastXC?color=blue&logo=open-source-initiative" alt="MIT License">
  </a>
  <a href="https://github.com/your-org/FastXC/stargazers">
    <img src="https://img.shields.io/github/stars/your-org/FastXC?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/your-org/FastXC/issues">
    <img src="https://img.shields.io/github/issues/your-org/FastXC?logo=github" alt="Open issues">
  </a>
  <a href="https://github.com/your-org/FastXC/pulls">
    <img src="https://img.shields.io/github/issues-pr/your-org/FastXC?logo=github" alt="Open pull requests">
  </a>
  <img src="https://img.shields.io/github/last-commit/your-org/FastXC?logo=git" alt="last commit">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia" alt="CUDA >=11.8">
</p>


* 切换语言 / Switch language: [English](README.md)

---

# FastXC
**面向 1/9 分量环境噪声的 CPU‑GPU 异构互相关流水线**

FastXC 将 **CUDA‑C 计算核心** 与 **Python 调度层** 融为一体，
实现从 SAC 波形到叠加 NCF 的高速自动化处理。

> **注意**：当前 *PWS* 与 *tf‑PWS* 内核需要 **≥ 20 GB** 显存，> 显存较小请仅用线性叠加或缩短窗口。

---

## 🚩 主要特性
| 类别 | 亮点 |
|------|------|
| **速度** | CUDA 加速 `sac2spec_ultra`、`xc_fast`、`RotateNCF` |
| **灵活** | 支持单台阵/双台阵、单分量/九分量 |
| **叠加** | 线性、**PWS**、**tf‑PWS** |
| **自动** | 单一 `.ini` 配置 + `FastXCPipeline` 步骤模式 |
| **整洁** | 正则检索 SAC，无需改名，自动列表/拼接 |

---

## 🌱 安装指南
```bash
git clone https://github.com/your-org/FastXC
cd FastXC

# 1. 编译 CUDA/C 可执行文件
cd cuda_c_src && make -j && cd ..

# 2. Python 环境
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. 生成并编辑模板配置
python -m fastxc --generate-template
vim template_config.ini.copy

# 4. 运行
python -m fastxc template_config.ini.copy
```

**编译小贴士**  
GPU 架构不同请修改 `cuda_c_src/Makefile`：
```
export ARCH=sm_89   # 例如 RTX 4090 → 8.9
```
可用脚本检测：
```bash
utils/check_gpu/compile.sh && utils/check_gpu/check_gpu
```

---

## 🏗 目录结构快览
<details>
<summary>点击展开</summary>

```text
cuda_c_src/          CUDA 源码
fastxc/              Python 调度
  cmd_generator/     生成命令
  cmd_deployer/      派发执行
  list_generator/    生成文件列表
  utils/             工具函数
config/              *.ini 示例
run.py               最小入口
```
</details>

---

## ⚙️ 流水线与模式
FastXC 顺序执行 8 个步骤：

| # | 步骤 | 作用 | 默认 |
|---|------|------|------|
| 1 | GenerateFilter     | 设计滤波器                         | ALL |
| 2 | OrganizeSAC        | 分组 SAC 文件                      | ALL |
| 3 | Sac2Spec           | GPU FFT → `segspec/`               | ALL |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                    | ALL |
| 5 | ConcatenateNcf     | 合并日尺度 NCF                     | ALL |
| 6 | Stack              | 线性 / PWS / tf‑PWS 叠加          | ALL |
| 7 | Rotate             | ZNE ↔ RT / HR 旋转                | ALL |
| 8 | Sac2Dat            | 可选 SAC→DAT                       | SKIP |

使用 `StepMode` 细粒度控制：
```python
from fastxc import StepMode, FastXCPipeline
pipe = FastXCPipeline("my.ini")
pipe.run({{
    "Rotate": StepMode.SKIP
}})
```

---

## 📝 配置文件速查
```ini
[array_info1]
sac_dir = /data/array1
pattern = {{home}}/{{YYYY}}/{{station}}.{{component}}.{{suffix}}
component_list = E,N,Z

[parameters]
output_dir = /data/out
delta      = 0.05
bands      = 0.1/0.5 0.5/1
max_lag    = 600
win_len    = 3600
```
完整中文注释版请运行 `python -m fastxc --generate-template`。

---

## 🖥 环境检查
```bash
nvidia-smi
nvcc --version
```

---

## ❓ 常见问题
<details><summary>展开</summary>

* **Windows 支持？** — 建议 WSL2，未做原生测试。  
* **性能瓶颈？** — I/O 往往主导总耗时，推荐 NVMe SSD。  
* **MULTI 为何缺 tf‑PWS？** — 计划迭代支持。  

</details>

---

## 📜 参考文献
* Wang 等 (2025)《High‑performance CPU‑GPU Heterogeneous Computing Method for 9‑Component Ambient Noise Cross‑Correlation》*Earthquake Research Advances*（in press）。  
* Bensen G.D. 等 (2007) *GJI* 169(3):1239‑1260。  
* Cupillard P. 等 (2011) *GJI* 184(3):1397‑1414。  
* Zhang Y. 等 (2018) *JGR* 123(9):8016‑8031。

---

© 2023‑2025 Your Name & Contributors — 使用 MIT 许可证
