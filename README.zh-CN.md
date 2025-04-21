<p align="center">
  <a href="https://github.com/wangkingh/FastXC/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/wangkingh/FastXC_ultra/ci.yml?branch=main&label=CI&logo=github" alt="CI 状态">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/wangkingh/FastXC_ultra?color=blue&logo=open-source-initiative" alt="MIT 许可证">
  </a>
  <a href="https://github.com/wangkingh/FastXC/stargazers">
    <img src="https://img.shields.io/github/stars/wangkingh/FastXC_ultra?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/wangkingh/FastXC/issues">
    <img src="https://img.shields.io/github/issues/wangkingh/FastXC_ultra?logo=github" alt="未解决 Issue">
  </a>
  <a href="https://github.com/wangkingh/FastXC/pulls">
    <img src="https://img.shields.io/github/issues-pr/wangkingh/FastXC_ultra?logo=github" alt="开放的 Pull Request">
  </a>
  <img src="https://img.shields.io/github/last-commit/wangkingh/FastXC_ultra?logo=git" alt="最后提交时间">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia" alt="CUDA ≥11.8">
</p>

* Switch language / 切换语言: [English](README.md)

---

# FastXC
**适用于背景噪声互相关（1 分量 & 9 分量）的高性能 CPU‑GPU 异构处理流程**

FastXC 通过 **CUDA‑C 内核** 与 **Python 控制器** 协同工作，将原始 SAC 波形快速转换为叠加后的噪声相关函数（NCF）。

> **注意**：当前 *PWS* 与 *tf‑PWS* 内核需要 **≥ 20 GB 显存** 的 GPU。  
> 显存较小的显卡仍可运行线性叠加或较短窗口。

---

## 🚩 主要特性
| 类别 | 亮点 |
|------|------|
| **速度** | 针对 `sac2spec_ultra`、`xc_fast`、`RotateNCF` 的 CUDA 加速内核 |
| **灵活性** | 支持单阵列 **或** 双阵列互相关，单分量 **或** 3×3=9 分量流程 |
| **叠加** | 线性、**PWS**、**tf‑PWS**（GPU 加速） |
| **自动化** | 单文件配置 (`.ini`) + `FastXCPipeline` 分步骤模式（`SKIP`、`CMD_ONLY` 等） |
| **清晰 I/O** | 基于正则的 SAC 检索、自动文件列表、可选合并与 DAT 输出 |

---

## 🌱 快速安装
```bash
git clone https://github.com/wangkingh/FastXC_ultra

cd utils

bash compile.sh

./chech_gpu

# 0. 查看 GPU 信息，根据输出修改 cuda_c_src/Makefile 中的 arch 选项

cd FastXC

# 1. 编译 CUDA/C 可执行文件
cd cuda_c_src && make veryclean && make

# 2. 检查 Python 环境
pip install numpy pandas scipy obspy ...

# 3. 生成模板配置或从仓库复制
python -m fastxc --generate-template
vim template_config.ini.copy   # 修改绝对路径

# 4. 在 run.py 中修改配置路径并运行完整流水线
python run.py
```

### 构建贴士
*更换 GPU 计算架构* — 编辑 `cuda_c_src/Makefile`
```
export ARCH=sm_89   # 例如 RTX 4090 → CC 8.9
```
检查方法：
```bash
utils/check_gpu/compile.sh && utils/check_gpu/check_gpu
```

---

## 🏗 目录结构一览
<details>
<summary>点击展开</summary>

```text
cuda_c_src/          CUDA 内核与 Makefile
  ├─ sac2spec_stable/ 将 sac 转为谱
  ├─ sac2spec_ultra/  使用额外滤波将 sac 转为谱
  ├─ xc/              计算谱间互相关
  ├─ ncf_pws/         叠加 NCF
  ├─ rotate/          将 NCF 从 ENZ 坐标旋转到 RTZ
  ├─ Makefile
fastxc/              Python 调度器
  ├─ cmd_generator/   生成 *.cmds.txt
  ├─ cmd_deployer/    派发命令
  ├─ list_generator/  生成文件列表
  └─ utils/           配置解析、滤波设计等
bin/
  ├─ sac2spec_stable
  ├─ sac2spec_ultra
  ├─ xc_fast
  ├─ ncf_pws
  ├─ RotateNCF
utils/
  ├─ check_gpu/     检测 GPU 能力
  ├─ extract.py     互相关后将 “bigsac” 拆分为 sac
  ├─ GPU_vs_CPU.png

config/              示例 *.ini
run.py               最小启动脚本
```
</details>

---

## ⚙️ 流水线 & 运行模式
`run.py` 按顺序执行八个步骤：

| # | 步骤 | 目的 |
|---|------|------|
| 1 | GenerateFilter     | 设计 FIR/IIR (`filter.txt`)          |
| 2 | OrganizeSAC        | 按台站/时间整理 SAC                 |
| 3 | Sac2Spec           | GPU FFT → `segspec/` 频谱           |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                     |
| 5 | ConcatenateNcf     | 合并每日 NCF                        |
| 6 | Stack              | 线性 / PWS / tf‑PWS                 |
| 7 | Rotate             | ZNE ↔ RTZ                           |
| 8 | Sac2Dat            | 可选 SAC → DAT 转换                 |

每步可选择处理模式：

| 值 | 模式名 | 说明 |
|----|--------|------|
| 1 | SKIP        | 不执行              |
| 2 | PREPARE     | 仅生成目录或文件列表 |
| 3 | CMD_ONLY    | 仅生成命令行        |
| 4 | DEPLOY_ONLY | 仅派发命令（不生成） |
| 5 | ALL         | 执行以上全部动作     |

使用 `StepMode` 覆盖：

```python
from fastxc import StepMode, FastXCPipeline
pipe = FastXCPipeline("my.ini")
pipe.run({
    "CrossCorrelation": StepMode.CMD_ONLY,
    "Rotate": StepMode.SKIP,
})
```

---

## 📝 配置速查（详细请见模板文件）
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
执行 `python -m fastxc --generate-template` 生成完整带注释模板。

---

## 🖥 环境检查
```bash
nvidia-smi
nvcc --version
```

---

## ℹ️ 常见问题
<details><summary>本仓库与 <a href="https://github.com/wangkingh/FastXC" target="_blank">FastXC</a> 有何区别？</summary>

1. **更高采样率支持**  
   *PWS* / *tf‑PWS* 的 CUDA 批处理实现已重写，可处理更高采样率数据。

2. **磁盘友好的 NCF 输出**  
   在 **xc** 阶段，各段 NCF 现在 **追加写入原文件尾部**，而非写入新文件。

3. **取消 “dual” 模式**  
   删除 *dual* 工作流。互相关与叠加在单一高性能模式运行，速度更快，但占用更多磁盘。

4. **时间戳过滤**  
   可提供时间列表文件，仅保留落在指定时间段的 SAC 文件。

5. **更清晰的配置**  
   配置文件重新设计：常用项前置，进阶参数放在 **advanced** 区域，便于自定义。

</details>

---

## 📒 更新日志
见 [Change Log](changelog.md)

## 📧 作者联系方式
如有问题、建议或想贡献代码，请在 [issue](https://github.com/wangkingh/FastXC/issues) 里讨论或提交 Pull Request。

直接联系作者：  
**Email:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

若本项目能助力您的研究，敬请告知，作者不胜荣幸！

## 🙏 致谢
衷心感谢中国科学技术大学、中国地震局地球物理研究所、中国地震局地震预测研究所以及中国科学院地质与地球物理研究所的同事们在本程序测试和试运行期间给予的 __重要支持__！

## 📜 参考文献
Wang 等 (2025). ["High-performance CPU-GPU Heterogeneous Computing Method for 9-Component Ambient Noise Cross-correlation."](https://doi.org/10.1016/j.eqrea.2024.100357) Earthquake Research Advances. In Press.

Bensen, G. D., 等 (2007). ["Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements."](https://dx.doi.org/10.1111/j.1365-246x.2007.03374.x) Geophysical Journal International 169(3): 1239‑1260.

Cupillard, P., 等 (2011). ["The one-bit noise correlation: a theory based on the concepts of coherent and incoherent noise."](https://doi.org/10.1111/j.1365-246X.2010.04923.x) Geophysical Journal International 184(3): 1397‑1414.

---

© 2023‑2025 Wang Jingxi & ChatGPT O3 — 采用 MIT 许可证
