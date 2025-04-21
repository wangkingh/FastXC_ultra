from typing import List, Dict
import os
import glob


# find min and max value in a string like '0.1/0.5 0.5/1.0 1.0/2.0'
def find_min_max_in_string(s):
    elements = s.split()

    numbers = []
    for element in elements:
        numbers.extend(map(float, element.split("/")))

    min_value = min(numbers)
    max_value = max(numbers)

    return f"{min_value}/{max_value}"


# generate sac2spec terminal cmd
def gen_sac2spec_cmd(
    component_num: int,
    sac2spec_cmd: str,
    parameter: Dict,
    gpu_num: int,
    cpu_count_per_thread: int,
    whiten: str,
    skip_step: str,
) -> List[str]:
    """
    Generate sac 2 spec commands for both 1 and 3 components
    """
    win_len = parameter["win_len"]
    shift_len = parameter["shift_len"]
    whiten_types = {"OFF": 0, "BEFORE": 1, "AFTER": 2, "BOTH": 3}
    cuda_whiten = whiten
    cuda_whiten = whiten_types.get(cuda_whiten, 2)
    normalize_types = {"OFF": 0, "RUN-ABS-MF": 1, "ONE-BIT": 2, "RUN-ABS": 3}
    cuda_normalize = parameter["normalize"]
    cuda_normalize = normalize_types.get(cuda_normalize, 1)
    band_info = parameter["bands"]
    whiten_band = find_min_max_in_string(band_info)

    # iterate over the gpu_dir in the work_dir
    output_dir = parameter["output_dir"]
    sac_spec_list_dir = os.path.join(output_dir, "sac_spec_list")

    # the filter file is in the output_dir
    filter_file = os.path.join(output_dir, "filter.txt")

    param_set = [
        "-L",
        str(win_len),
        "-S",
        str(shift_len),
        "-W",
        str(cuda_whiten),
        "-N",
        str(cuda_normalize),
        "-F",
        whiten_band,
        "-Q",
        str(skip_step),
        "-B",
        filter_file,
        "-U",
        str(gpu_num),
        "-T",
        str(cpu_count_per_thread),
    ]

    cmd_list = []
    sac_lists = glob.glob(os.path.join(sac_spec_list_dir, "sac_list_*.txt"))
    gpu_num = len(sac_lists)
    for sac_list in sac_lists:
        info = os.path.basename(sac_list).split(".")[0]
        gpu_id = info.split("_")[-1]
        spec_list = os.path.join(sac_spec_list_dir, f"spec_list_{gpu_id}.txt")
        local_param_set = [sac2spec_cmd, "-I", sac_list, "-O", spec_list]
        local_param_set += ["-C", str(component_num)]
        local_param_set += ["-G", str(gpu_id)]
        local_param_set += param_set
        cmd = " ".join(local_param_set)
        cmd_list.append(cmd)

    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    os.makedirs(cmd_list_dir, exist_ok=True)
    cmd_file = os.path.join(cmd_list_dir, "sac2spec_cmds.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(cmd_list))
    return cmd_list
