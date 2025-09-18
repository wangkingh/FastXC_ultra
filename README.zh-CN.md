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

<!-- 切换语言 / Switch language: [English](README.md) -->

---

# FastXC Ultra
**面向单/双阵列（1 & 9 分量）环境噪声互相关的高性能 CPU‑GPU 流水线**  

FastXC Ultra 以 **Python 控制器** 协调 **CUDA‑C 内核**，将原始 SAC 波形在几分钟内堆叠成噪声互相关函数（NCF）——昔日的“数小时”现在只需“数分钟”。

> **注意**：GPU 叠加（PWS / tf‑PWS）仍需 **≥ 20 GB 显存/卡**。  
> 显存更小的显卡可使用线性叠加或更短窗口。

---

## 🚩 关键特性
| 分类 | 亮点 |
|------|------|
| **速度** | `sac2spec_ultra`、`xc_fast`、`ncf_pws` 等 CUDA 加速内核 |
| **灵活** | 支持单阵列 **或** 双阵列，单分量 **或** 3×3 = 9 分量流程 |
| **叠加** | 线性、**PWS**、**tf‑PWS** 皆享 GPU 加速 |
| **自动化** | 单个 INI 配置 + `FastXCPipeline` 分步模式（`SKIP`、`CMD_ONLY` …） |
| **干净 I/O** | 正则匹配 SAC、自动文件列表、增量写入 NCF |

---

## 🌱 快速安装
```bash
# 0. 克隆仓库
git clone https://github.com/wangkingh/FastXC_ultra
cd FastXC_ultra

# 1. 编译 CUDA/C 可执行文件（必需）
cd cuda_c_src && make veryclean && make    # 如需更改架构，编辑 Makefile 中 ARCH

# 2. 修改配置文件
vim my.ini

# 3. 修改主控脚本 run.py
vim run.py 修改配置文件路径

# 4. 运行 run.py
python run.py
```

### 架构提示
在 `cuda_c_src/Makefile` 中修改计算能力，例如：
```make
export ARCH = sm_89      # RTX 4090 → CC 8.9
```
验证：
```bash
utils/check_gpu/compile.sh && utils/check_gpu/check_gpu
```

---

## 🏗 目录结构
<details>
<summary>点击展开</summary>

```text
cuda_c_src/          CUDA 内核 + Makefile
  ├─ sac2spec_ultra/   SAC → 频谱
  ├─ xc/               频谱 × 频谱
  ├─ ncf_pws/          PWS / tf‑PWS 叠加
  ├─ rotate/           ENZ ↔ RTZ 旋转
  └─ Makefile
fastxc/              Python 调度器
  ├─ cmd_generator/
  ├─ cmd_deployer/
  ├─ list_generator/
  └─ utils/
bin/                 编译后可执行文件
utils/               GPU 检测、绘图脚本等
config/              示例 *.ini
run.py               最小启动器
```
</details>

---

## ⚙️ 流水线与模式
`run.py` 依次执行 8 个步骤：

| # | 步骤 | 目的 |
|---|------|------|
| 1 | GenerateFilter     | FIR/IIR 设计（生成 `filter.txt`）        |
| 2 | OrganizeSAC        | 按台站/时间整理 SAC                      |
| 3 | Sac2Spec           | GPU FFT → `segspec/` 频谱               |
| 4 | CrossCorrelation   | GPU XC → `ncf/`                         |
| 5 | ConcatenateNcf     | 合并日级 NCF                            |
| 6 | Stack              | 线性 / PWS / tf‑PWS                     |
| 7 | Rotate             | ENZ ↔ RTZ                               |
| 8 | Sac2Dat            | 可选：SAC → DAT                         |

每步可选模式：

| 模式 | 行为 |
|------|------|
| `SKIP`        | 跳过 |
| `PREPARE`     | 仅生成目录 / 文件列表 |
| `CMD_ONLY`    | 只生成终端命令 |
| `DEPLOY_ONLY` | 执行命令（不重新生成） |
| `ALL`         | 以上全部 |

```python
from fastxc import StepMode, FastXCPipeline
pipe = FastXCPipeline("my.ini")
pipe.run({
    "CrossCorrelation": StepMode.CMD_ONLY,
    "Rotate": StepMode.SKIP,
})
```

---

## 📝 配置速查
下例节选自 **2025‑05‑02** 版本；完整带注释模板请执行：
```bash
python -m fastxc --generate-template
```

```ini
[array1]                         ; ---- 数据源 #1 ----
sac_dir       = ./               ; 根目录
pattern       = {home}/{*}/{*}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
time_start    = 2017-09-01 00:00:00
time_end      = 2017-09-30 01:00:00
component_list= E,N,Z            ; ENZ 顺序
time_list     = NONE
sta_list      = NONE

[array2]                         ; ---- 数据源 #2（可选） ----
sac_dir       = NONE
pattern       = {home}/{YYYY}/{station}_{component}_{JJJ}.{suffix}
component_list= Z                ; 单分量示例

[preprocess]                     ; ---- 预处理 ----
win_len     = 3600               ; 秒
shift_len   = 3600
delta       = 0.1                ; 采样间隔 → 10 Hz
normalize   = RUN-ABS-MF         ; 滑动绝对值 + 中值滤波
bands       = 0.1/0.5 0.5/1 1/2  ; 频带（空格分隔）
whiten      = BEFORE             ; XC 前预白化
skip_step   = -1                 ; -1=不跳窗口；或 0,1,2...

[xcorr]                          ; ---- 互相关 ----
max_lag     = 100                ; 秒
write_mode  = APPEND             ; 增量写 NCF
write_segment = False            ; 不保存每段 NC
distance_range = -1/50000        ; km，限制台站距
azimuth_range  = -1/360          ; °，限制方位
source_info_file = NONE

[stack]                          ; ---- 叠加 ----
stack_flag     = 100             ; 1=线性，0=PWS，0=tf‑PWS
sub_stack_size = 1
source_info_file = NONE

[executables]                    ; ---- 可执行文件 ----
sac2spec = /path/to/bin/sac2spec_ultra
xc       = /path/to/bin/xc_multi_channel
stack    = /path/to/bin/ncf_pws
rotate   = /path/to/bin/RotateNCF

[device]                         ; ---- 硬件 ----
gpu_list      = 0,1,2,3
gpu_task_num  = 1,1,1,1
gpu_mem_info  = 40,40,40,40     ; GiB
cpu_count     = 100

[storage]                        ; ---- 输出 ----
output_dir = ./
overwrite   = True
clean_ncf   = True

[debug]                          ; ---- 调试 ----
dry_run        = False
log_file_path  = NONE
```

---

## 🖥 环境自检
```bash
nvidia-smi                   # 驱动需支持 CUDA 11.8+
nvcc --version               # 工具链与驱动一致
python -m fastxc --doctor    # 内置诊断脚本
```

---

## ℹ️ 常见问题
<details><summary>只跑单阵列怎么办？</summary>

将 `[array2].sac_dir = NONE` 即可自动切换为单阵列模式。
</details>

<details><summary>如何跳过旋转？</summary>

可省略 `[rotate]` 小节，或运行时指定：
```python
pipe.run({"Rotate": StepMode.SKIP})
```
</details>

---

## 📒 更新日志
详见 [CHANGELOG.md](CHANGELOG.md)。

## 📧 联系方式
如有问题或建议，请 [提交 Issue](https://github.com/wangkingh/FastXC_ultra/issues) 或邮件联系：  
**Email:** <wkh16@mail.ustc.edu.cn>

---

© 2023‑2025 Wang Jingxi with ChatGPT O3 — 基于 **MIT License** 许可。
