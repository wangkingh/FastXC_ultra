from typing import Dict, List, Tuple
from pandas import Timestamp
import numpy as np
import logging
import os
import sys

logger = logging.getLogger(__name__)


def distribute_tasks(
    gpu_list: List, gpu_memory: List, num_tasks: int
) -> Dict[int, int]:
    """
    Function to distribute tasks among available GPUs based on their memory size.
    """
    assert len(gpu_list) == len(
        gpu_memory
    ), "gpu_list and gpu_memory must have the same length!"

    # Calculate the number of tasks to assign to each GPU based on their memory size
    total_memory = sum(gpu_memory)
    gpu_tasks = {}
    for i, gpu_id in enumerate(gpu_list):
        ratio = gpu_memory[i] / total_memory
        tasks_for_this_gpu = int(np.floor(num_tasks * ratio))
        gpu_tasks[gpu_id] = tasks_for_this_gpu

    # Calculate the number of tasks already assigned
    assigned_tasks = sum(gpu_tasks.values())

    # If there are remaining tasks, assign them to the GPU with the largest memory
    remaining_tasks = num_tasks - assigned_tasks
    if remaining_tasks > 0:
        idx_of_largest = np.argmax(gpu_memory)
        largest_gpu = gpu_list[idx_of_largest]
        gpu_tasks[largest_gpu] += remaining_tasks

    # Assertion to check that all tasks have been assigned
    assert (
        sum(gpu_tasks.values()) == num_tasks
    ), "Not all tasks were correctly assigned!"
    return gpu_tasks


def build_sac_spec_pair(
    station: str,
    time: Timestamp,
    components: List[str],
    paths: List[str],
    spec_dir: str,
    array_id: str,
    network: str = "VV",            # <── 新增
) -> Dict[str, List[str]]:
    """
    given station, time, components, paths, spec_dir, array_id, return a dict containing sac and spec paths
    """
    time_str = time.strftime("%Y.%j.%H%M")
    sac_paths = []
    spec_paths = []

    for component, sac_path in zip(components, paths):
        spec_name = f"{network}.{station}.{time_str}.{component}.segspec"
        array_flag = f"array{array_id}"
        full_spec_path = os.path.join(spec_dir, array_flag, time_str, spec_name)

        sac_paths.append(sac_path)
        spec_paths.append(full_spec_path)

    return {"sac": sac_paths, "spec": spec_paths}


def build_sac_spec_pairs_for_group(
    seis_file_group: Dict[Tuple[str, Timestamp], Dict[str, List[str]]],
    spec_dir: str,
    array_id: str,
    component_flag: int,
    placeholder_net: str = "VV",
) -> List[Dict[str, List[str]]]:
    """
    for each group, build sac_spec_pairs
    """
    results = []
    if not seis_file_group:
        return results

    # ------- 判断 key 形状 --------
    sample_key = next(iter(seis_file_group))
    has_network = len(sample_key) == 3
    
    for key, file_info_dict in seis_file_group.items():
        if has_network:
            station, time, network = key      # 解包 3 元
        else:
            station, time = key               # 解包 2 元
            network = placeholder_net         # 填占位
            
        components = file_info_dict["component"]
        paths = file_info_dict["path"]
        if len(components) != component_flag:
            logger.warning(
                f"[Group {array_id}] station={station} time={time}, "
                f"found {len(components)} components, expected {component_flag}."
            )
            continue

        sac_spec_pair = build_sac_spec_pair(
            station=station,
            time=time,
            components=components,
            paths=paths,
            spec_dir=spec_dir,
            array_id=array_id,
            network=network, 
        )
        results.append(sac_spec_pair)

    return results


def gen_sac2spec_list(
    files_group1,
    files_group2,
    gpu_list,
    gpu_memory,
    component_list1,
    component_list2,
    output_dir,
):
    """
    generate sac2spec list
    """

    if not files_group1:
        logger.warning("files_group1 is empty.Stop\n"+'-'*80+'\n')
        sys.exit(1)
    if not files_group2:
        logger.warning("files_group2 is empty.\n"+'-'*80+'\n')
    if not files_group1 and not files_group2:
        logger.warning("No SAC files found in both groups, skip writing specs.\n")
        return

    sac_spec_list_dir = os.path.join(output_dir, "sac_spec_list")
    spec_dir = os.path.join(output_dir, "segspec")
    os.makedirs(sac_spec_list_dir, exist_ok=True)

    pairs_1 = []
    pairs_2 = []
    if files_group1:
        pairs_1 = build_sac_spec_pairs_for_group(
            files_group1, spec_dir, 1, len(component_list1)
        )
    if files_group2:
        pairs_2 = build_sac_spec_pairs_for_group(
            files_group2, spec_dir, 2, len(component_list2)
        )
    sac_spec_pairs = pairs_1 + pairs_2

    sac_spec_pair_num = len(sac_spec_pairs)
    gpu_tasks = distribute_tasks(gpu_list, gpu_memory, sac_spec_pair_num)

    pair_index = 0
    gpu_pair_map = {}
    for gpu, task_num in gpu_tasks.items():
        gpu_sac_pairs = sac_spec_pairs[pair_index : pair_index + task_num]
        gpu_pair_map[gpu] = gpu_sac_pairs
        pair_index += task_num

    for gpu, allocated_pairs in gpu_pair_map.items():
        sac_list_file = os.path.join(sac_spec_list_dir, f"sac_list_{gpu}.txt")
        spec_list_file = os.path.join(sac_spec_list_dir, f"spec_list_{gpu}.txt")

        with open(sac_list_file, "w") as f_sac, open(spec_list_file, "w") as f_spec:
            for pair in allocated_pairs:
                components_num = len(pair["sac"])
                if components_num != len(component_list1):
                    logger.warning(
                        f"Components number {components_num} is not equal to {len(component_list1)}"
                    )
                for component_index in range(components_num):
                    sac_file = pair["sac"][component_index]
                    spec_file = pair["spec"][component_index]
                    f_sac.write(sac_file + "\n")
                    f_spec.write(spec_file + "\n")

    return
