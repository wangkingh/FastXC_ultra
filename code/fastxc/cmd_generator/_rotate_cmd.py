import os


def gen_rotate_cmd(rotate_cmd: str, output_dir: str) -> bool:
    """
    iterate all enz_list.txt and rtz_list.txt files in the rotate_list directory,
    and generate a list of commands to rotate the seismograms.
    The command is like: my_rotate_program -I <enz_list.txt> -O <rtz_list.txt>
    """
    rotate_list_dir = os.path.join(output_dir, "rotate_list")
    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    os.makedirs(cmd_list_dir, exist_ok=True)

    # 1. iterate all labels in the rotate_list directory
    for label in os.listdir(rotate_list_dir):
        label_cmds = []
        label_dir = os.path.join(rotate_list_dir, label)
        if not os.path.isdir(label_dir):
            continue  # skip non-directory

        # 2. iterate all station pairs in the label directory
        for sta_pair in os.listdir(label_dir):
            sta_pair_dir = os.path.join(label_dir, sta_pair)
            if not os.path.isdir(sta_pair_dir):
                continue

            # 3. get the input and output file paths
            inlist = os.path.join(sta_pair_dir, "enz_list.txt")
            outlist = os.path.join(sta_pair_dir, "rtz_list.txt")
            if not (os.path.isfile(inlist) and os.path.isfile(outlist)):
                continue

            # 4. generate the command
            # e.g.：my_rotate_program -I <enz_list.txt> -O <rtz_list.txt>
            cmd = f"{rotate_cmd} -I {inlist} -O {outlist}"
            label_cmds.append(cmd)

        # 5. write the command list to a file
        cmd_file = os.path.join(cmd_list_dir, f"rotate_cmds_{label}.txt")
        with open(cmd_file, "w") as f:
            f.write("\n".join(label_cmds))

    return True
