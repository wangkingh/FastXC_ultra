from typing import List
import os, glob
import logging

logger = logging.getLogger(__name__)


def gen_stack_cmd(
    stack_exe: str,
    output_dir: str,
    stack_flag: str,
) -> List[str]:
    """
    Generate commands for stack, including GPU arguments.

    :param stack_exe: path to the stack executable
    :param output_dir: main output directory
    :param stack_flag: 'S' option for -S (e.g. "100", "110", "011", etc.)
    :return: A list of command strings
    """
    cmd_list = []
    # stack_dir: the actual directory where final results go
    stack_dir = os.path.join(output_dir, "stack")
    # where we place the generated command list
    cmd_list_dir = os.path.join(output_dir, "cmd_list")

    ncf_dir = os.path.join(output_dir, "ncf")
    if not os.path.exists(ncf_dir):
        logger.warning(f"ncf directory {ncf_dir} does not exist. ")
        return []
    all_big_sac = glob.glob(os.path.join(ncf_dir, "*.bigsac"))
    os.makedirs(stack_dir, exist_ok=True)
    os.makedirs(cmd_list_dir, exist_ok=True)

    for big_sac in all_big_sac:
        # 组装命令行增加 -G 和 -U 参数
        cmd = f"{stack_exe} " f"-I {big_sac} " f"-O {stack_dir} " f"-S {stack_flag}"
        cmd_list.append(cmd)

    # 写入到文件
    cmd_list_path = os.path.join(cmd_list_dir, "stack_cmds.txt")
    with open(cmd_list_path, "w") as f:
        f.write("\n".join(cmd_list))

    return cmd_list
