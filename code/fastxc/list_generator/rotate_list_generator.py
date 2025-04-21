from typing import List
import logging
import os
from itertools import product

logger = logging.getLogger(__name__)


def prepare_enz_dirs(stack_flag: str, output_dir: str):
    linear_flag = int(stack_flag[0])  # first digit -> linear
    pws_flag = int(stack_flag[1])  # second digit -> pws
    tfpws_flag = int(stack_flag[2])  # third digit -> tfpws

    enz_dirs = []
    if linear_flag:
        enz_dir = os.path.join(output_dir, "stack", "linear")
        enz_dirs.append(enz_dir)
    if pws_flag:
        enz_dir = os.path.join(output_dir, "stack", "pws")
        enz_dirs.append(enz_dir)
    if tfpws_flag:
        enz_dir = os.path.join(output_dir, "stack", "tfpws")
        enz_dirs.append(enz_dir)

    return enz_dirs


def _gen_rotate_list(
    enz_dir, label, component_list_1, component_list_2, output_dir
) -> bool:
    """
    Prepare the input and output file lists for the rotation from ENZ to RTZ.
    """
    rtz_ncf_dir = os.path.join(output_dir, "stack", f"rtz_{label}")
    rotate_list_dir = os.path.join(output_dir, "rotate_list")

    mapping = {component_list_1[i]: val for i, val in enumerate(["E", "N", "Z"])}
    mapping.update({component_list_2[i]: val for i, val in enumerate(["E", "N", "Z"])})

    ENZ_pair_order = ["E-E", "E-N", "E-Z", "N-E", "N-N", "N-Z", "Z-E", "Z-N", "Z-Z"]
    RTZ_pair_order = ["R-R", "R-T", "R-Z", "T-R", "T-T", "T-Z", "Z-R", "Z-T", "Z-Z"]

    for sta_pair in os.listdir(enz_dir):
        sta_pair_path = os.path.join(enz_dir, sta_pair)
        enz_sac_num = len(os.listdir(sta_pair_path))
        if enz_sac_num != 9:
            continue
        enz_group = {}
        for component_pair in product(component_list_1, component_list_2):
            component1, component2 = component_pair
            component_info = f"{mapping[component1]}-{mapping[component2]}"
            fname = f"{sta_pair}.{component1}-{component2}.ncf.sac"
            file_path = os.path.join(sta_pair_path, fname)
            enz_group.update({component_info: file_path})

        rotate_dir = os.path.join(rotate_list_dir, f"{label}", sta_pair)
        os.makedirs(rotate_dir, exist_ok=True)

        in_list = os.path.join(rotate_dir, "enz_list.txt")
        out_list = os.path.join(rotate_dir, "rtz_list.txt")

        # Write the input stack file paths
        with open(in_list, "w") as f:
            enz_group = {key: enz_group[key] for key in ENZ_pair_order}
            f.write("\n".join(enz_group.values()))

        # Write the output RTZ file paths
        with open(out_list, "w") as f:
            for component_pair in RTZ_pair_order:
                outpath = os.path.join(
                    rtz_ncf_dir, sta_pair, f"{sta_pair}.{component_pair}.ncf.sac"
                )
                f.write(outpath + "\n")
    return True


def gen_rotate_list(
    component_list1: List, component_list2: List, stack_flag: str, output_dir: str
) -> bool:
    """
    Generate the rotate list for the given seismic array configuration.
    """
    # get all stack directories for ENZ
    enz_dirs = prepare_enz_dirs(stack_flag, output_dir)

    if not enz_dirs:
        logger.warning("No ENZ directories found for generating rotate lists.")
        return False

    # iterate over all ENZ directories
    for enz_dir in enz_dirs:
        label = os.path.basename(enz_dir)
        logger.info(f"Generating rotate list for ENZ dir: {enz_dir}, label: {label}")
        _gen_rotate_list(
            enz_dir=enz_dir,
            label=label,
            component_list_1=component_list1,
            component_list_2=component_list2,
            output_dir=output_dir,
        )
    return True
