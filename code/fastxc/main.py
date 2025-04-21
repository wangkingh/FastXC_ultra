# fastxc/main.py
import os
import shutil
import logging
import sys
from difflib import get_close_matches

# 假设外部模块路径：
from .utils import parse_and_check_ini_file
from .utils import design_filter
from .utils import orgnize_sacfile

from .list_generator import gen_sac2spec_list
from .list_generator import gen_xc_list
from .list_generator import gen_rotate_list

from .cmd_generator import gen_sac2spec_cmd
from .cmd_generator import gen_xc_cmd
from .cmd_generator import gen_stack_cmd
from .cmd_generator import gen_rotate_cmd

from .cmd_deployer import sac2spec_deployer
from .cmd_deployer import xc_deployer
from .cmd_deployer import rotate_deployer
from .cmd_deployer import pws_deployer

from .utils import sac2dat_deployer
from .utils import concatenate_ncf

logger = logging.getLogger(__name__)


# ========== 1. CONFIG 对象 ==========
class Config:
    """负责解析配置文件并存储所有必要的上下文信息。"""

    def __init__(self, config_path):
        (
            self.array_info1,
            self.array_info2,
            self.parameters,
            self.executables,
            self.device_info,
            self.adv_proc,
            self.adv_storage,
            self.adv_debug,
            self.future_options,
        ) = parse_and_check_ini_file(config_path)

        # SAC 组织结果
        self.stas1 = None
        self.stas2 = None
        self.times1 = None
        self.times2 = None
        self.files_group1 = None
        self.files_group2 = None

    @property
    def is_double_array(self) -> bool:
        return self.array_info2.get("sac_dir", "NONE") != "NONE"


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
        mode 的取值见 StepMode，可在子类里根据需要细化。
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
    if clean:
        logger.info(f"Cleaning up directory: {path}")
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


# ========== 3. 各具体 STEP ==========


class GenerateFilterStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="GenerateFilter")

    def execute(self, mode: str):
        """
        对于滤波器设计，只需要一个“准备”阶段即可，
        并没有“生成命令、部署命令”这种概念。
        所以这里直接根据 _prepare_if() 判断是否执行。
        """
        logger.debug(f"[GenerateFilter] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping filter generation.")
            return

        if self._prepare_if(mode):
            # do the filter generation
            delta = self.config.parameters["delta"]
            bands = self.config.parameters["bands"]
            output_dir = self.config.parameters["output_dir"]
            output_path = os.path.join(output_dir, "filter.txt")
            design_filter(delta, bands, output_path)
            logger.info("Filter file generated.\n")
        else:
            logger.info("GenerateFilter: nothing to do in current mode.")


class OrganizeSacfileStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="OrganizeSAC")

    def execute(self, mode: str):
        logger.debug(f"[OrganizeSacfile] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping SAC file organizing.")
            return

        # 同理，“只生成”、“只部署”在这里都指的是“组织SAC文件”本身没有命令行可部署
        if self._prepare_if(mode):
            cpu_count = self.config.device_info["cpu_count"]
            stas1, stas2, times1, times2, fg1, fg2 = orgnize_sacfile(
                self.config.array_info1, self.config.array_info2, cpu_count
            )
            self.config.stas1 = stas1
            self.config.stas2 = stas2
            self.config.times1 = times1
            self.config.times2 = times2
            self.config.files_group1 = fg1
            self.config.files_group2 = fg2
            logger.info("SAC file organization done.\n")
        else:
            logger.info("OrganizeSacfile: nothing to do in current mode.\n")


class Sac2SpecStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Sac2Spec")

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

    def _generate_sac2spec_list(self):
        if not self.config.files_group1 and not self.config.files_group2:
            logger.warning(
                "files_group1 and files_group2 is empty. Possibly forgot to organize SAC files?\n"
            )
            return
        output_dir = self.config.parameters["output_dir"]
        comp1 = self.config.array_info1["component_list"]
        comp2 = self.config.array_info2["component_list"]
        gpu_list = self.config.device_info["gpu_list"]
        gpu_memory = self.config.device_info["gpu_mem_info"]

        gen_sac2spec_list(
            self.config.files_group1,
            self.config.files_group2,
            gpu_list,
            gpu_memory,
            comp1,
            comp2,
            output_dir,
        )
        logger.info("sac2spec list generated.\n")

    def _generate_sac2spec_cmd(self):
        component_num = len(self.config.array_info1["component_list"])
        sac2spec_cmd = self.config.executables["sac2spec"]
        cpu_count = self.config.device_info["cpu_count"]
        gpu_num = len(self.config.device_info["gpu_list"]) or 1
        cpu_count_per_thread = cpu_count // gpu_num
        whiten_place = self.config.adv_proc["whiten"]
        skip_step = self.config.adv_proc["skip_step"]

        gen_sac2spec_cmd(
            component_num,
            sac2spec_cmd,
            self.config.parameters,
            gpu_num,
            cpu_count_per_thread,
            whiten_place,
            skip_step,
        )
        logger.info("sac2spec command generated.\n")

    def _deploy_sac2spec_cmd(self):
        output_dir = self.config.parameters["output_dir"]
        segspec_dir = os.path.join(output_dir, "segspec")
        cmd_list_file = os.path.join(output_dir, "cmd_list", "sac2spec_cmds.txt")
        sac_spec_list_dir = os.path.join(output_dir, "sac_spec_list")
        log_file_path = self.config.adv_debug["log_file_path"]
        dry_run = self.config.adv_debug["dry_run"]

        sac2spec_deployer(
            cmd_list_file, sac_spec_list_dir, segspec_dir, log_file_path, dry_run
        )
        logger.info("sac2spec commands deployed and executed.\n"+'_'*80+'\n')


class CrossCorrelationStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="CrossCorrelation")

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

    def _generate_xc_list(self):
        out_dir = self.config.parameters["output_dir"]
        seg_dir = os.path.join(out_dir, "segspec")
        xc_list_dir = os.path.join(out_dir, "xc_list")
        num_thread = self.config.device_info["cpu_count"]

        gen_xc_list(seg_dir, xc_list_dir, num_thread)
        logger.info("XC list generated.\n")

    def _generate_xc_cmd(self):
        single_array = not self.config.is_double_array
        output_dir = self.config.parameters["output_dir"]
        xc_list_dir = os.path.join(output_dir, "xc_list")
        xc_cmd_list = os.path.join(output_dir, "cmd_list", "xc_cmds.txt")
        ncf_dir = os.path.join(output_dir, "ncf")

        max_lag = self.config.parameters["max_lag"]
        dist_range = self.config.adv_proc["distance_range"]
        azimuth_range = self.config.adv_proc["azimuth_range"]
        cpu_num = self.config.device_info["cpu_count"]
        gpu_task_nums = self.config.device_info["gpu_task_num"]
        gpu_num_task_total = sum(gpu_task_nums)
        cpu_num_per_gpu = cpu_num // gpu_num_task_total
        source_file = self.config.future_options["source_info_file"]

        write_mode = self.config.adv_storage["write_mode"]
        exe_to_use = self.config.executables["xc"]

        gen_xc_cmd(
            single_array=single_array,
            xc_list_dir=xc_list_dir,
            xc_cmd_list=xc_cmd_list,
            xc_exe=exe_to_use,
            ncf_dir=ncf_dir,
            cclength=max_lag,  # 原先叫 max_lag，这里要改成 cclength
            dist_range=dist_range,  # 原先叫 distance_range，这里改成 dist_range
            azimuth_range=azimuth_range,  # 原先叫 azimuth_range，这里改成 az_range
            cpu_count=cpu_num_per_gpu,  # 原先叫 cpu_num_per_gpu_thread，这里改成 cpu_count
            srcinfo_file=source_file,  # 原先叫 source_info_file，这里改成 srcinfo_file
            write_mode=write_mode,
        )

        logger.info("XC command generated.\n")

    def _deploy_xc_cmd(self):
        out_dir = self.config.parameters["output_dir"]
        xc_cmd_list = os.path.join(out_dir, "cmd_list", "xc_cmds.txt")
        gpu_list = self.config.device_info["gpu_list"]
        gpu_task_num = self.config.device_info["gpu_task_num"]
        log_file_path = self.config.adv_debug["log_file_path"]
        dry_run = self.config.adv_debug["dry_run"]

        xc_deployer(xc_cmd_list, gpu_list, gpu_task_num, log_file_path, dry_run)
        logger.info("XC command deployed and executed.\n"+'_'*80+'\n')


class StackStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Stack")

    def execute(self, mode: str):
        logger.debug(f"[Stack] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping stack step.")
            return

        # if self._prepare_if(mode):
        #     self._generate_stack_cmd()

        if self._generate_cmd_if(mode):
            self._generate_stack_cmd()

        if self._deploy_if(mode):
            self._deploy_stack_cmd()

    def _generate_stack_cmd(self):
        stack_exe = self.config.executables["stack"]
        output_dir = self.config.parameters["output_dir"]
        stack_flag = self.config.parameters["stack_flag"]
        gen_stack_cmd(stack_exe, output_dir, stack_flag)
        logger.info("Stack command generated.\n")

    def _deploy_stack_cmd(self):
        write_mode = self.config.adv_storage["write_mode"]
        if write_mode == "AGGREGATE":
            logger.warning(
                "AGGREGATE mode has already stacked the ncf in xc process.PWS and tf-PWS result is not reliable here"
            )
        out_dir = self.config.parameters["output_dir"]

        stack_cmd_file = os.path.join(out_dir, "cmd_list", "stack_cmds.txt")
        log_file_path = self.config.adv_debug["log_file_path"]
        dry_run = self.config.adv_debug["dry_run"]

        gpu_list = self.config.device_info["gpu_list"]
        gpu_task_num = self.config.device_info["gpu_task_num"]
        cpu_count = self.config.device_info["cpu_count"]
        pws_deployer(stack_cmd_file, gpu_list, gpu_task_num, log_file_path,cpu_count, dry_run)
        logger.info("Stack commands deployed.\n"+'_'*80+'\n')


class RotateStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Rotate")

    def execute(self, mode: str):
        logger.debug(f"[Rotate] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping rotation step.")
            return

        if self._prepare_if(mode):
            # 旋转也需要先生成列表
            self._generate_rotate_list()

        if self._generate_cmd_if(mode):
            self._generate_rotate_cmd()

        if self._deploy_if(mode):
            self._deploy_rotate_cmd()

    def _generate_rotate_list(self):
        out_dir = self.config.parameters["output_dir"]
        stack_flag = self.config.parameters["stack_flag"]
        comp1 = self.config.array_info1["component_list"]
        comp2 = self.config.array_info2["component_list"]
        double_array = self.config.is_double_array

        if double_array:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning(
                    "Double array rotation needs 3 components in each array. Skipping."
                )
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single array rotation needs 3 components. Skipping.")
                return

        gen_rotate_list(comp1, comp2, stack_flag, out_dir)
        logger.info("Rotate list generated.")

    def _generate_rotate_cmd(self):
        out_dir = self.config.parameters["output_dir"]
        comp1 = self.config.array_info1["component_list"]
        comp2 = self.config.array_info2["component_list"]
        double_array = self.config.is_double_array

        if double_array:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning(
                    "Double array rotation needs 3 components in each array. Skipping."
                )
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single array rotation needs 3 components. Skipping.")
                return
        rotate_exe = self.config.executables["rotate"]
        gen_rotate_cmd(rotate_exe, out_dir)
        logger.info("Rotate command generated.")

    def _deploy_rotate_cmd(self):
        out_dir = self.config.parameters["output_dir"]
        comp1 = self.config.array_info1["component_list"]
        comp2 = self.config.array_info2["component_list"]
        double_array = self.config.is_double_array

        if double_array:
            if len(comp1) != 3 or len(comp2) != 3:
                logger.warning(
                    "Double array rotation needs 3 components in each array. Skipping."
                )
                return
        else:
            if len(comp1) != 3:
                logger.warning("Single array rotation needs 3 components. Skipping.")
                return
        cpu_count = self.config.device_info["cpu_count"]
        log_file_path = self.config.adv_debug["log_file_path"]
        dry_run = self.config.adv_debug["dry_run"]

        rotate_deployer(out_dir, cpu_count, log_file_path, dry_run)
        logger.info("Rotate commands deployed and executed.\n"+'_'*80+'\n')


class Sac2DatStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="Sac2Dat")

    def execute(self, mode: str):
        logger.debug(f"[Sac2Dat] Mode = {mode}")

        if mode == StepMode.SKIP:
            logger.info("Skipping sac2dat step.")
            return

        if self._deploy_if(mode):
            sac2dat_deployer(self.config.parameters)
            logger.info("SAC2DAT deployed & executed.\n"+'_'*80+'\n')


class ConcatenateNcfStep(Step):
    def __init__(self, config: Config):
        super().__init__(config, name="ConcatenateNcf")
        self.config = config

    def execute(self, mode: str):
        logger.debug(f"[ConcatenateNcf] Mode = {mode}")
        if mode == StepMode.SKIP:
            logger.info("Skipping ConcatenateNcf step.")
            return

        if self._deploy_if(mode):
            output_dir = self.config.parameters["output_dir"]
            cpu_count = self.config.device_info["cpu_count"]
            concatenate_ncf(output_dir, cpu_count)
            logger.info("ConcatenateNcf deployed & executed.\n"+'_'*80+'\n')


# ========== 4. 管线主类：FastXCPipeline ==========
class FastXCPipeline:
    """
    通过 steps_config 指定每个步骤的执行模式。
    steps_config 是一个 dict，如:
    {
        "GenerateFilter": "ALL",
        "OrganizeSAC": "PREPARE",
        "Sac2Spec": "SKIP",
        ...
    }
    你可以在这里配置自己需要的步骤和执行模式。
    """

    def __init__(self, config_path: str):
        self.config = Config(config_path)

        # 默认的步骤顺序
        self.ordered_steps = [
            GenerateFilterStep(self.config),
            OrganizeSacfileStep(self.config),
            Sac2SpecStep(self.config),
            CrossCorrelationStep(self.config),
            ConcatenateNcfStep(self.config),
            StackStep(self.config),
            RotateStep(self.config),
            Sac2DatStep(self.config),
        ]

    def run(self, steps_config: dict = None):
        """
        执行整个流程，并根据 steps_config 决定每一步怎么做。
        如果 steps_config 未提供某个 step 的配置，默认当成 StepMode.ALL。

        steps_config 的示例:
        {
          "GenerateFilter": StepMode.ALL,
          "OrganizeSAC": StepMode.ALL,
          "Sac2Spec": StepMode.PREPARE_ONLY,
          "CrossCorrelation": StepMode.CMD_ONLY,
          "ConcatenateNcf": StepMode.DEPLOY_ONLY,
          "Stack": StepMode.SKIP,
          "Rotate": StepMode.ALL
        }
        """
        if steps_config is None:
            steps_config = {}
        

        # 1. 收集所有合法的 step 名称
        valid_step_names = [step.name for step in self.ordered_steps]

        # 2. 对用户传进来的 steps_config 做一次“近邻检查+纠正”
        corrected_steps_config = {}
        for user_key, user_mode in steps_config.items():
            if user_key in valid_step_names:
                # 完全匹配，直接保留
                corrected_steps_config[user_key] = user_mode
            else:
                # 尝试模糊匹配
                suggestions = get_close_matches(
                    user_key, valid_step_names, n=1, cutoff=0.6
                )
                if suggestions:
                    close_match = suggestions[0]
                    logger.error(
                        f"Unrecognized step name '{user_key}'. "
                        f"Did you mean '{close_match}'?  Using '{close_match}' for mode '{user_mode}'."
                    )
                    sys.exit(1)
                    corrected_steps_config[close_match] = user_mode
                else:
                    logger.error(
                        f"Unrecognized step name '{user_key}'. No close match found; this entry will be ignored."
                    )
                    sys.exit(1)

        # 3. 用“纠正后”的 steps_config 来决定各步骤如何执行
        for step in self.ordered_steps:
            # 如果纠正后 dict 里依然没包含当前 step.name，则默认 ALL
            mode = corrected_steps_config.get(step.name, StepMode.ALL)
            logger.debug(f">>> Running step: {step.name} with mode: {mode}")
            step.execute(mode)

    @staticmethod
    def generate_template_config(output_path="template_config.ini.copy"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "template.ini")
        shutil.copy2(template_path, output_path)
        logger.info(f"Template configuration file copied to: {output_path}")
        print(f"Template configuration file has been copied to: {output_path}")
