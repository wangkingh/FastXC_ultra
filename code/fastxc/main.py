# fastxc/main.py
import os
from pathlib import Path
import shutil
import logging
import sys
from difflib import get_close_matches

# 假设外部模块路径：
from .config_parser import Config, ConfigError        # ← 改动：直接用新包

from .utils  import design_filter, orgnize_sacfile
from .list_generator import gen_sac2spec_list, gen_xc_list, gen_rotate_list
from .cmd_generator  import gen_sac2spec_cmd, gen_xc_cmd, gen_stack_cmd, gen_rotate_cmd
from .cmd_deployer   import sac2spec_deployer, xc_deployer, rotate_deployer, pws_deployer
from .utils          import sac2dat_deployer, concatenate_ncf

logger = logging.getLogger(__name__)



# ========== 2. STEP 基类 ==========
class StepMode:
    """
    定义几种执行模式，按需选用。也可以改成枚举 Enum。
    """

    SKIP = "SKIP"  # 不做任何事
    PREPARE_ONLY = "PREPARE"  # 只做“生成目录、生成文件列表”之类， 不执行/不部署
    CMD_ONLY = "CMD_ONLY"  # 只生成命令，不执行
    DEPLOY_ONLY = "DEPLOY"  # 只执行已有命令，不重新生成
    ALL = "ALL"  # 同时生成、也执行


class Step:
    """
    所有处理步骤的抽象基类。新增对 mode 的支持。
    子类在 execute() 中根据 mode 判断执行哪些子操作。
    """

    def __init__(self, config: Config, name: str):
        self.config = config
        self.name = name  # 给每个Step起一个名字，便于在外部控制。

    def execute(self, mode: str):
        """
        mode 的取值见 StepMode, 可在子类里根据需要细化。
        """
        raise NotImplementedError("Subclasses must implement 'execute(mode)' method.")

    def _prepare_if(self, mode: str):
        """检查 mode 是否允许做“准备工作”（如生成文件列表、创建目录等）"""
        return mode in (StepMode.PREPARE_ONLY, StepMode.ALL, StepMode.CMD_ONLY)

    def _deploy_if(self, mode: str):
        """检查 mode 是否允许做“部署执行”（如实际跑命令等）"""
        return mode in (StepMode.ALL, StepMode.DEPLOY_ONLY)

    def _generate_cmd_if(self, mode: str):
        """检查 mode 是否允许做“命令生成”"""
        return mode in (StepMode.ALL, StepMode.CMD_ONLY)


def prepare_directory(path: str, clean: bool = False):
    """负责创建或清理指定路径。"""
    p = Path(path).expanduser().resolve()
    if clean:
        logger.info(f"Cleaning up directory: {p}")
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


# ========== 3. 各具体 STEP ==========
class GenerateFilterStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="GenerateFilter")

    def execute(self, mode: str):
        """
        设计滤波器 (生成 filter.txt)。
        仅需“准备”阶段；无命令生成 / 部署。
        """
        logger.debug(f"[GenerateFilter] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping filter generation.")
            return

        # ★ 只有在 PREPARE / ALL / CMD_ONLY 时才干事
        if self._prepare_if(mode):
            delta      = self.config.preprocess.delta          # ★ 原 parameters["delta"]
            bands      = self.config.preprocess.bands          # ★ 原 parameters["bands"]
            output_dir: Path = self.config.storage.output_dir   # Path

            output_path = output_dir / "filter.txt"
            design_filter(delta, bands, output_path)
            logger.info("Filter file generated.\n")
        else:
            logger.info("GenerateFilter: nothing to do in current mode.")


class OrganizeSacfileStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="OrganizeSAC")

    # ------------------------------------------------------------------ #
    def execute(self, mode: str):
        logger.debug(f"[OrganizeSAC] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping SAC-file organizing.")
            return

        # 组织 SAC 文件没有 “命令生成 / 部署” 概念，只在 PREPARE / CMD_ONLY / ALL 时执行
        if self._prepare_if(mode):
            cpu_count = self.config.device.cpu_count
            array1 = self.config.array1
            array2 = getattr(self.config, "array2", None)   # None ⇒ 单阵列

            stas1, stas2, times1, times2, fg1, fg2 = orgnize_sacfile(
                array1, array2, cpu_count
            )

            # 结果挂到 Config 供后续 Step 使用
            self.config.stas1         = stas1
            self.config.stas2         = stas2
            self.config.times1        = times1
            self.config.times2        = times2
            self.config.files_group1  = fg1
            self.config.files_group2  = fg2

            logger.info("SAC-file organization done.\n")
        else:
            logger.info("OrganizeSAC: nothing to do in current mode.\n")


class Sac2SpecStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Sac2Spec")

    # ------------------------------------------------------------------ #
    #  总执行入口                                                         #
    # ------------------------------------------------------------------ #
    def execute(self, mode: str):
        logger.debug(f"[Sac2Spec] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping sac2spec step.")
            return

        # 1) 生成文件列表
        if self._prepare_if(mode):
            self._generate_sac2spec_list()

        # 2) 生成命令
        if self._generate_cmd_if(mode):
            self._generate_sac2spec_cmd()

        # 3) 部署执行
        if self._deploy_if(mode):
            self._deploy_sac2spec_cmd()

    # ------------------------------------------------------------------ #
    #  1. 生成待转换文件列表                                              #
    # ------------------------------------------------------------------ #
    def _generate_sac2spec_list(self):
        if not self.config.files_group1 and not self.config.files_group2:
            logger.warning(
                "files_group1 / files_group2 is empty - did OrganizeSAC run?\n"
            )
            return

        output_dir: Path = self.config.storage.output_dir
        comp1      = self.config.array1.component_list                     # ★
        comp2      = (
            self.config.array2.component_list if hasattr(self.config, "array2") else []
        )                                                                  # ★
        gpu_list   = self.config.device.gpu_list                           # ★
        gpu_memory = self.config.device.gpu_mem_info                       # ★

        gen_sac2spec_list(
            self.config.files_group1,
            self.config.files_group2,
            gpu_list,
            gpu_memory,
            comp1,
            comp2,
            str(output_dir),
        )
        logger.info("sac2spec list generated.\n")

    # ------------------------------------------------------------------ #
    #  2. 生成 sac2spec 命令                                             #
    # ------------------------------------------------------------------ #
    def _generate_sac2spec_cmd(self):
        # ---------- 核心参数 ------------------------------------------ #
        comp_num        = len(self.config.array1.component_list)
        sac2spec_exe    = self.config.executables.sac2spec                 # Path | str
        out_dir         = self.config.storage.output_dir             # Path

        pp              = self.config.preprocess                       # 预处理 dataclass
        win_len         = pp.win_len
        shift_len       = pp.shift_len
        normalize       = pp.normalize
        bands           = pp.bands
        whiten_place    = pp.whiten
        skip_step       = pp.skip_step                              # 允许 "0,5,10,-1" 或 "0/5/10/-1"

        # ---------- 资源参数 ------------------------------------------ #
        cpu_total       = self.config.device.cpu_count
        gpu_num         = max(1, len(self.config.device.gpu_list))
        cpu_per_thread  = cpu_total // gpu_num

        # ---------- 调用新 API ---------------------------------------- #
        gen_sac2spec_cmd(
            component_num  = comp_num,
            sac2spec_exe   = sac2spec_exe,
            output_dir     = out_dir,
            win_len        = win_len,
            shift_len      = shift_len,
            normalize      = normalize,
            bands          = bands,
            gpu_num        = gpu_num,
            cpu_per_thread = cpu_per_thread,
            whiten         = whiten_place,
            skip_step      = skip_step,
        )
        logger.info("sac2spec command generated.\n")

    # ------------------------------------------------------------------ #
    #  3. 部署执行                                                      #
    # ------------------------------------------------------------------ #
    def _deploy_sac2spec_cmd(self):
        output_dir: Path = self.config.storage.output_dir                 # ★ Path
        segspec_dir       = output_dir / "segspec"
        cmd_list_file     = output_dir / "cmd_list" / "sac2spec_cmds.txt"
        sac_spec_list_dir = output_dir / "sac_spec_list"

        log_file = self.config.debug.log_file_path                         # ★
        dry_run  = self.config.debug.dry_run                               # ★

        sac2spec_deployer(
            cmd_list_file, sac_spec_list_dir, segspec_dir, log_file, dry_run
        )
        logger.info("sac2spec commands deployed and executed.\n" + "_" * 80 + "\n")


class CrossCorrelationStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="CrossCorrelation")

    # ------------------------------------------------------------------ #
    #  主入口                                                             #
    # ------------------------------------------------------------------ #
    def execute(self, mode: str):
        logger.debug(f"[CrossCorrelation] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping cross-correlation step.")
            return

        if self._prepare_if(mode):
            self._generate_xc_list()

        if self._generate_cmd_if(mode):
            self._generate_xc_cmd()

        if self._deploy_if(mode):
            self._deploy_xc_cmd()

    # ------------------------------------------------------------------ #
    #  1. 生成待相关文件列表                                              #
    # ------------------------------------------------------------------ #
    def _generate_xc_list(self):
        out_dir: Path  = self.config.storage.output_dir      # 已是 Path
        seg_dir        = out_dir / "segspec"
        xc_list_dir    = out_dir / "xc_list"
        num_threads  = self.config.device.cpu_count          # device_info["cpu_count"]

        gen_xc_list(seg_dir, xc_list_dir, num_threads)
        logger.info("XC list generated.\n")
    
    # ------------------------------------------------------------------ #
    #  2. 生成互相关命令                                                 #
    # ------------------------------------------------------------------ #
    def _generate_xc_cmd(self):
        single_array = not self.config.is_double_array 

        out_dir: Path  = self.config.storage.output_dir
        xc_list_dir    = out_dir / "xc_list"
        xc_cmd_list    = out_dir / "cmd_list" / "xc_cmds.txt"
        ncf_dir        = out_dir / "ncf"

        max_lag       = self.config.xcorr.max_lag                 # parameters["max_lag"]
        dist_range    = self.config.xcorr.distance_range          # adv_proc["distance_range"]
        az_range      = self.config.xcorr.azimuth_range           # adv_proc["azimuth_range"]

        cpu_total     = self.config.device.cpu_count
        gpu_task_num  = self.config.device.gpu_task_num
        gpu_tasks     = max(1, sum(gpu_task_num))
        cpu_per_gpu   = cpu_total // gpu_tasks

        srcinfo_file  = self.config.xcorr.source_info_file        # future_options["source_info_file"]

        save_seg = self.config.xcorr.write_segment                  # bool
        write_mode    = self.config.xcorr.write_mode              # adv_storage["write_mode"]
        xc_exe        = self.config.executables.xc                      # executables["xc"]

        # gen_xc_cmd 旧签名保持不变，这里仅把字段名映射
        gen_xc_cmd(
            single_array   = single_array,
            xc_list_dir    = xc_list_dir,
            xc_cmd_list    = xc_cmd_list,
            xc_exe         = xc_exe,
            ncf_dir        = ncf_dir,
            cclength       = max_lag,
            dist_range     = dist_range,
            azimuth_range  = az_range,
            cpu_count      = cpu_per_gpu,
            srcinfo_file   = srcinfo_file,
            write_mode     = write_mode,
            save_segment = save_seg, 
        )
        logger.info("XC command generated.\n")

    # ------------------------------------------------------------------ #
    #  3. 部署执行                                                      #
    # ------------------------------------------------------------------ #
    def _deploy_xc_cmd(self):
        out_dir: Path   = self.config.storage.output_dir
        xc_cmd_list     = out_dir / "cmd_list" / "xc_cmds.txt"

        gpu_list       = self.config.device.gpu_list
        gpu_task_num   = self.config.device.gpu_task_num

        log_file_path  = self.config.debug.log_file_path
        dry_run        = self.config.debug.dry_run

        xc_deployer(xc_cmd_list, gpu_list, gpu_task_num, log_file_path, dry_run)
        logger.info("XC command deployed and executed.\n" + "_" * 80 + "\n")


class StackStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Stack")

    # ------------------------------------------------------------------ #
    def execute(self, mode: str):
        logger.debug(f"[Stack] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping stack step.")
            return

        if self._generate_cmd_if(mode):
            self._generate_stack_cmd()

        if self._deploy_if(mode):
            self._deploy_stack_cmd()

    # ------------------------------------------------------------------ #
    #  1. 生成 stack 命令                                                #
    # ------------------------------------------------------------------ #
    def _generate_stack_cmd(self):
        stack_exe   = self.config.executables.stack
        output_dir  = self.config.storage.output_dir
        stack_flag  = self.config.stack.stack_flag
        sub_size    = self.config.stack.sub_stack_size      # ← 取 INI 里的值

        gen_stack_cmd(
            stack_exe   = stack_exe,
            output_dir  = output_dir,
            stack_flag  = stack_flag,
            sub_stack_size = sub_size,                      # ← 传入
        )
        logger.info("Stack command generated.\n")

    # ------------------------------------------------------------------ #
    #  2. 部署执行                                                      #
    # ------------------------------------------------------------------ #
    def _deploy_stack_cmd(self):
        write_mode = self.config.storage.write_mode        # ★ adv_storage["write_mode"]
        if write_mode == "AGGREGATE":
            logger.warning(
                "AGGREGATE mode already stacked NCF in XC process; "
                "PWS and TF-PWS result may be unreliable here."
            )

        out_dir: Path = self.config.storage.output_dir          # Path
        stack_cmd_file = out_dir / "cmd_list" / "stack_cmds.txt"  # Path

        log_file_path = self.config.debug.log_file_path    # ★ adv_debug
        dry_run       = self.config.debug.dry_run          # ★

        gpu_list      = self.config.device.gpu_list        # ★ device_info
        gpu_task_num  = self.config.device.gpu_task_num    # ★
        cpu_count     = self.config.device.cpu_count       # ★

        pws_deployer(
            stack_cmd_file,
            gpu_list,
            gpu_task_num,
            log_file_path,
            cpu_count,
            dry_run,
        )
        logger.info("Stack commands deployed.\n" + "_" * 80 + "\n")



class RotateStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Rotate")

    # ------------------------------------------------------------------ #
    def execute(self, mode: str):
        logger.debug(f"[Rotate] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping rotation step.")
            return

        if self._prepare_if(mode):
            self._generate_rotate_list()

        if self._generate_cmd_if(mode):
            self._generate_rotate_cmd()

        if self._deploy_if(mode):
            self._deploy_rotate_cmd()

    # ------------------------------------------------------------------ #
    #  1. 生成旋转-NCF 列表                                              #
    # ------------------------------------------------------------------ #
    def _generate_rotate_list(self):
        out_dir     = self.config.storage.output_dir            # ★
        stack_flag  = self.config.stack.stack_flag              # ★
        comp1       = self.config.array1.component_list         # ★
        comp2       = (
            self.config.array2.component_list if hasattr(self.config, "array2") else []
        )                                                       # ★
        double_arr  = self.config.is_double_array

        if double_arr:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning("Double-array rotation needs 3 components each – skipping.")
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single-array rotation needs 3 components – skipping.")
                return

        gen_rotate_list(comp1, comp2, stack_flag, out_dir)
        logger.info("Rotate list generated.")

    # ------------------------------------------------------------------ #
    #  2. 生成旋转命令                                                   #
    # ------------------------------------------------------------------ #
    def _generate_rotate_cmd(self):
        out_dir    = self.config.storage.output_dir
        comp1      = self.config.array1.component_list
        comp2      = (
            self.config.array2.component_list if hasattr(self.config, "array2") else []
        )
        double_arr = self.config.is_double_array

        if double_arr:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning("Double-array rotation needs 3 components each - skipping.")
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single-array rotation needs 3 components - skipping.")
                return

        rotate_exe = self.config.executables.rotate                    # ★
        gen_rotate_cmd(rotate_exe, str(out_dir))
        logger.info("Rotate command generated.")

    # ------------------------------------------------------------------ #
    #  3. 部署执行                                                       #
    # ------------------------------------------------------------------ #
    def _deploy_rotate_cmd(self):
        out_dir    = self.config.storage.output_dir
        comp1      = self.config.array1.component_list
        comp2      = (
            self.config.array2.component_list if hasattr(self.config, "array2") else []
        )
        double_arr = self.config.is_double_array

        if double_arr:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning("Double-array rotation needs 3 components each – skipping.")
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single-array rotation needs 3 components – skipping.")
                return

        cpu_cnt     = self.config.device.cpu_count               # ★
        log_path    = self.config.debug.log_file_path            # ★
        dry_run     = self.config.debug.dry_run                  # ★

        rotate_deployer(out_dir, cpu_cnt, log_path, dry_run)
        logger.info("Rotate commands deployed & executed.\n" + "_" * 80 + "\n")

# -------------------------------------------------------------------------- #
#  SAC ➜ DAT 批量转换                                                        #
# -------------------------------------------------------------------------- #
class Sac2DatStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Sac2Dat")

    def execute(self, mode: str):
        logger.debug(f"[Sac2Dat] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping sac2dat step.")
            return

        if self._deploy_if(mode):
            # sac2dat_deployer 旧接口期望 “parameters dict”
            # 这里只给它最常用的三个键；如还需要更多可再补
            params = {
                "output_dir": self.config.storage.output_dir,      # ★
                "delta":      self.config.preprocess.delta,           # ★
                "bands":      self.config.preprocess.bands,           # ★
            }
            sac2dat_deployer(params)
            logger.info("SAC2DAT deployed & executed.\n" + "_" * 80 + "\n")


# -------------------------------------------------------------------------- #
#  拼接 NCF (concatenate)                                                    #
# -------------------------------------------------------------------------- #
class ConcatenateNcfStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="ConcatenateNcf")

    def execute(self, mode: str):
        logger.debug(f"[ConcatenateNcf] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping ConcatenateNcf step.")
            return

        if self._deploy_if(mode):
            out_dir   = self.config.storage.output_dir          # ★
            cpu_count = self.config.device.cpu_count            # ★

            concatenate_ncf(out_dir, cpu_count)
            logger.info("ConcatenateNcf deployed & executed.\n" + "_" * 80 + "\n")


# ========== 4. Pipeline 主控类：FastXCPipeline ==========
class FastXCPipeline:
    """
    steps_config 用于为每一步指定执行模式::

        {
            "GenerateFilter": StepMode.ALL,
            "OrganizeSAC":   StepMode.PREPARE_ONLY,
            "Sac2Spec":      StepMode.SKIP,
            ...
        }
    若某步未给出，则默认 StepMode.ALL。
    """

    # ------------------------------------------------------------------ #
    def __init__(self, ini_path: str):
        # ---------- 加载 & 校验 ----------
        try:
            self.cfg = Config(ini_path)          # ★ 直接使用新的 Config
            self.cfg.validate_all()
        except ConfigError as e:
            logger.error(e)
            sys.exit(1)

        # ---------- 默认步骤顺序 ----------
        self.ordered_steps = [
            GenerateFilterStep(self.cfg),        # ★ self.cfg
            OrganizeSacfileStep(self.cfg),
            Sac2SpecStep(self.cfg),
            CrossCorrelationStep(self.cfg),
            ConcatenateNcfStep(self.cfg),
            StackStep(self.cfg),
            RotateStep(self.cfg),
            Sac2DatStep(self.cfg),
        ]

    # ------------------------------------------------------------------ #
    def run(self, steps_config: dict | None = None):
        if steps_config is None:
            steps_config = {}

        # 1) 合法 step 名称
        valid_names = [step.name for step in self.ordered_steps]

        # 2) 检查 / 纠正用户输入
        corrected: dict[str, str] = {}
        for user_key, user_mode in steps_config.items():
            if user_key in valid_names:
                corrected[user_key] = user_mode
                continue

            # 模糊匹配
            suggestion = get_close_matches(user_key, valid_names, n=1, cutoff=0.6)
            if suggestion:
                close = suggestion[0]
                logger.error(
                    f"Unrecognized step '{user_key}'. Did you mean '{close}'? "
                    f"→ aborting."
                )
            else:
                logger.error(f"Unrecognized step '{user_key}'. No close match – aborting.")
            sys.exit(1)

        # 3) 依次执行
        for step in self.ordered_steps:
            mode = corrected.get(step.name, StepMode.ALL)
            logger.debug(f">>> Running step: {step.name}  |  mode: {mode}")
            step.execute(mode)

    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_template_config(output_path: str = "template_config.ini.copy"):
        """把包内 template.ini 复制到指定位置，便于用户起手修改。"""
        current_dir   = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "template.ini")
        shutil.copy2(template_path, output_path)

        logger.info(f"Template configuration file copied to: {output_path}")
        print(f"Template configuration file has been copied to: {output_path}")
