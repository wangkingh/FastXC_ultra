import sys
import os
import configparser
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def check_path_exists(path, description):
    if path != "NONE" and not os.path.exists(path):
        logger.error(f"{description} not found: {path}")
        sys.exit(1)
    return True


def check_time_format(timestr, desc):
    try:
        datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.error(f"Invalid date format for {desc}: '{timestr}'")
        sys.exit(1)


def check_time_list_file(time_list_path):
    """
    check if the time_list file exists and each line is in the correct format
    """
    if time_list_path == "NONE":
        return

    if not os.path.exists(time_list_path):
        logger.error(f"time_list file '{time_list_path}' does not exist.")
        sys.exit(1)

    with open(time_list_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            # 如果是空行或者以#开头的行，都跳过不做检查
            if not line or line.startswith('#'):
                continue
            check_time_format(line, desc=f"line {line_num} in '{time_list_path}'")

def check_skip_step(skip_step_str: str) -> str:
    """
    check the skip_step string for correct format.
    """
    # 1. split by '/'
    parts = skip_step_str.split("/")
    if len(parts) < 1:
        logger.error(
            f"skip_step='{skip_step_str}' was not split properly by '/' or set to '-1'"
        )
        sys.exit(1)
    # 2.the last part must be '-1'
    if parts[-1] != "-1":
        logger.error(f"skip_step='{skip_step_str}' should end with '-1'.")
        sys.exit(1)
    # 3.
    for p in parts:
        try:
            int(p)
        except ValueError:
            logger.error(
                f"skip_step='{skip_step_str}' include  non-integer value '{p}'."
            )
            sys.exit(1)
    return skip_step_str


def convert_type(key, value):
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower in ["true"]:
            return True
        elif val_lower in ["false"]:
            return False

    # 再针对不同 key 做定制化转换
    if key in [
        "npts",
        "win_len",
        "shift_len",
        "cpu_count",
        "max_lag",
    ]:
        return int(value)

    elif key in ["delta"]:
        return float(value)

    elif key in ["gpu_list", "gpu_task_num", "gpu_mem_info"]:
        return [int(x) for x in value.split(",")]

    elif key in ["component_list"]:
        return [x.strip() for x in value.split(",")]

    return value


def load_config_file(file_path):
    """
    read the configuration file and return the configparser object.
    """
    if not os.path.exists(file_path):
        logger.error(f"Configuration file '{file_path}' does not exist.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(file_path)

    if not config.sections():
        logger.error("No sections found in configuration file.")
        sys.exit(1)

    return config


def parse_sections(config):
    """
    parse config file and return the corresponding dictionary.
      - array_info1
      - array_info2
      - parameters
      - executables
      - device_info
      - advanced_process
      - advanced_storage
      - advanced_debug
      - future_options
    return a tuple of the above dictionaries.
    """
    try:
        array_info1 = {
            key: convert_type(key, config.get("array_info1", key, fallback="NONE"))
            for key in config["array_info1"]
        }
        array_info2 = {
            key: convert_type(key, config.get("array_info2", key, fallback="NONE"))
            for key in config["array_info2"]
        }
        parameters = {
            key: convert_type(key, config.get("parameters", key, fallback="NONE"))
            for key in config["parameters"]
        }
        executables = {
            key: config.get("executables", key, fallback="NONE")
            for key in config["executables"]
        }
        device_info = {
            key: convert_type(key, config.get("device_info", key, fallback="NONE"))
            for key in config["device_info"]
        }
        advanced_process = {
            key: convert_type(key, config.get("advanced_process", key, fallback="NONE"))
            for key in config["advanced_process"]
        }
        advanced_storage = {
            key: convert_type(key, config.get("advanced_storage", key, fallback="NONE"))
            for key in config["advanced_storage"]
        }
        advanced_debug = {
            key: convert_type(key, config.get("advanced_debug", key, fallback="NONE"))
            for key in config["advanced_debug"]
        }
        future_options = {
            key: convert_type(key, config.get("Future_options", key, fallback="NONE"))
            for key in config["Future_options"]
        }
    except KeyError as e:
        logger.error(f"Missing section or key in configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration properly: {e}")
        sys.exit(1)

    return (
        array_info1,
        array_info2,
        parameters,
        executables,
        device_info,
        advanced_process,
        advanced_storage,
        advanced_debug,
        future_options,
    )


def check_array_info(array_info1, array_info2):
    """
    check the array_info1 and array_info2 dictionaries for necessary keys and values.
    """
    # 1) sac_dir path
    check_path_exists(array_info1["sac_dir"], "SAC File directory for array1")
    array_info1["sac_dir"] = os.path.abspath(array_info1["sac_dir"])
    if array_info2["sac_dir"] != "NONE":
        array_info2["sac_dir"] = os.path.abspath(array_info2["sac_dir"])
        check_path_exists(array_info2["sac_dir"], "SAC File directory for array2")

    # 2) sta_list
    if array_info1.get("sta_list", "NONE") != "NONE":
        check_path_exists(array_info1["sta_list"], "Station list for array1")
    if array_info2.get("sta_list", "NONE") != "NONE":
        check_path_exists(array_info2["sta_list"], "Station list for array2")

    # 3) time format
    check_time_format(array_info1["time_start"], "array_info1:time_start")
    check_time_format(array_info1["time_end"], "array_info1:time_end")

    # 3.1) time_list 文件检查 (array_info1)
    time_list_1 = array_info1.get("time_list", "NONE")
    check_time_list_file(time_list_1)

    if array_info2["sac_dir"] != "NONE":
        check_time_format(array_info2["time_start"], "array_info2:time_start")
        check_time_format(array_info2["time_end"], "array_info2:time_end")

        # 3.2) time_list 文件检查 (array_info2)
        time_list_2 = array_info2.get("time_list", "NONE")
        check_time_list_file(time_list_2)

        # 4) component_list
        if len(array_info1["component_list"]) != len(array_info2["component_list"]):
            logger.error(
                "component_list in array_info1 and array_info2 must be the same length."
            )
            sys.exit(1)
    logger.debug("Array information checks passed.")


def check_executables(executables):
    """
    ensure the executables paths exist.
    """
    for key, path in executables.items():
        check_path_exists(path, f"Executable for {key}")
    logger.debug("Executable paths check passed.")


def check_parameters(parameters):
    """
    check the parameters dictionary for necessary keys and values.
    check output_dir and normalize value.
    """
    normalize_val = parameters.get("normalize", "OFF")
    if normalize_val not in ["OFF", "RUN-ABS", "ONE-BIT", "RUN-ABS-MF"]:
        logger.error(
            f"Invalid normalize value '{normalize_val}' in [parameters]. "
            f"Should be one of ['OFF','RUN-ABS','ONE-BIT','RUN-ABS-MF']."
        )
    # check and abs the output_dir path
    output_dir = parameters.get("output_dir", "NONE")
    if output_dir == "NONE":
        logger.error("output_dir not specified in [parameters].")
        sys.exit(1)
    else:
        output_dir = os.path.abspath(output_dir)
        parameters["output_dir"] = output_dir
    logger.debug("Parameter checks passed.")


def check_device_info(device_info):
    """
    CHECK the device_info dictionary for necessary keys and values.
    """

    # 1) cpu_count
    cpu_count = device_info.get("cpu_count", 1)
    if isinstance(cpu_count, str):
        cpu_count = int(cpu_count)
    if cpu_count < 1:
        logger.error("'cpu_count' must be >= 1.")
        sys.exit(1)
    device_info["cpu_count"] = cpu_count

    # 2) gpu_list, gpu_task_num, gpu_mem_info
    gpu_list = device_info.get("gpu_list", "NONE")
    gpu_task_num = device_info.get("gpu_task_num", [])
    gpu_mem_info = device_info.get("gpu_mem_info", [])

    if not isinstance(gpu_list, list):
        logger.error(f"'gpu_list' must be a list, got '{gpu_list}'.")
        sys.exit(1)
    if not isinstance(gpu_task_num, list):
        logger.error(f"'gpu_task_num' must be a list, got '{gpu_task_num}'.")
        sys.exit(1)
    if not isinstance(gpu_mem_info, list):
        logger.error(f"'gpu_mem_info' must be a list, got '{gpu_mem_info}'.")
        sys.exit(1)

    if not (len(gpu_list) == len(gpu_task_num) == len(gpu_mem_info)):
        logger.error(
            f"'gpu_list', 'gpu_task_num', 'gpu_mem_info' must have the same length, "
            f"got lengths: {len(gpu_list)}, {len(gpu_task_num)}, {len(gpu_mem_info)}."
        )
        sys.exit(1)

    for i, gpu_id in enumerate(gpu_list):
        if gpu_id < 0:
            logger.error(f"'gpu_list' contains negative GPU ID: {gpu_id}.")
            sys.exit(1)
    for i, task_num in enumerate(gpu_task_num):
        if task_num < 0:
            logger.error(
                f"'gpu_task_num' contains invalid task number: {task_num} (GPU index {i})."
            )
            sys.exit(1)
    for i, mem_val in enumerate(gpu_mem_info):
        if mem_val <= 0:
            logger.error(
                f"'gpu_mem_info' must be positive, got {mem_val} (GPU index {i})."
            )
            sys.exit(1)

    if len(set(gpu_list)) != len(gpu_list):
        logger.error("'gpu_list' has duplicate GPU IDs.")
        sys.exit(1)

    device_info["gpu_list"] = gpu_list
    device_info["gpu_task_num"] = gpu_task_num
    device_info["gpu_mem_info"] = gpu_mem_info
    logger.debug("Device information checks passed.")


def check_advanced_proc_options(adv_opts):
    """
    check and parse the advanced pre-processing options dictionary for necessary keys and values.
      - whiten
      - skip_step
      - distance_range
      - azimuth_range
      - tfpws_band_limited
    """
    # 1) whiten
    whiten_val = adv_opts.get("whiten", "OFF")
    if whiten_val not in ["OFF", "BEFORE", "AFTER", "BOTH"]:
        logger.error(
            f"Invalid 'whiten' value '{whiten_val}'. "
            f"Should be one of ['OFF','BEFORE','AFTER','BOTH']."
        )
        sys.exit(1)
    adv_opts["whiten"] = whiten_val

    # 2) skip_step
    raw_skip_val = adv_opts.get("skip_step", "-1")  # 默认 -1 表示无跳步
    skip_step_str = check_skip_step(str(raw_skip_val))  # 验证并返回合法值
    adv_opts["skip_step"] = skip_step_str

    # 3) ditance_range
    dist_str = adv_opts.get("distance_range", "NONE").strip()
    parts = dist_str.split("/")
    if len(parts) != 2:
        logger.error(f"'distance_range' must be 'lower/upper', got '{dist_str}'")
    lower_str, upper_str = parts
    try:
        lower_val = float(lower_str)
        upper_val = float(upper_str)
    except ValueError:
        logger.error(f"'distance_range' has non-float values: '{dist_str}'")
        sys.exit(1)
    if lower_val > upper_val:
        logger.error(
            f"distance_range lower='{lower_val}' cannot exceed upper='{upper_val}'."
        )
        sys.exit(1)
    adv_opts["distance_range"] = dist_str

    # 4) ditance_range
    azimuth_str = adv_opts.get("azimuth_range", "NONE").strip()
    parts = azimuth_str.split("/")
    if len(parts) != 2:
        logger.error(f"'azimuth_range' must be 'lower/upper', got '{azimuth_str}'")
    lower_str, upper_str = parts
    try:
        lower_val = float(lower_str)
        upper_val = float(upper_str)
    except ValueError:
        logger.error(f"'distance_range' has non-float values: '{azimuth_str}'")
        sys.exit(1)
    if lower_val > upper_val:
        logger.error(
            f"azimuth_range lower='{lower_val}' cannot exceed upper='{upper_val}'."
        )
        sys.exit(1)
    adv_opts["azimuth_range"] = azimuth_str

    # 4) tfpws_band_limited
    tfpws_flag = adv_opts.get("tfpws_band_limited", False)
    if not isinstance(tfpws_flag, bool):
        logger.error(f"'tfpws_band_limited' must be bool, got '{tfpws_flag}'")
        sys.exit(1)
    adv_opts["tfpws_band_limited"] = tfpws_flag
    logger.debug("Advanced processing options checks passed.")


def check_advanced_storage_options(adv_opts):
    """
    check and parse the advanced storage options dictionary for necessary keys and values.
      - save_disk
      - clean_ncf
      - overwrite_mode
    """
    # 1) write_mode
    write_mode_val = adv_opts.get("write_mode", True)
    if write_mode_val not in ["APPEND", "AGGREGATE"]:
        logger.error(
            f"'write_mode' must be 'APPEND' or 'AGGREGATE', got '{write_mode_val}'"
        )
    adv_opts["write_mode"] = write_mode_val

    # 1) clean_ncf
    clean_ncf_val = adv_opts.get("clean_ncf", True)
    if isinstance(clean_ncf_val, str):
        clean_ncf_val = clean_ncf_val.lower() in ["true", "1", "yes"]
    adv_opts["clean_ncf"] = clean_ncf_val

    # 2) overwrite_mode
    overwrite_mode = adv_opts.get("overwrite_mode", True)
    if isinstance(overwrite_mode, str):
        overwrite_mode = overwrite_mode.lower() in ["true", "1", "yes"]
    if not isinstance(overwrite_mode, bool):
        logger.error(f"'overwrite_mode' must be bool, got '{overwrite_mode}'")
    adv_opts["overwrite_mode"] = overwrite_mode

    logger.debug("Advanced storage options checks passed.")


def check_advanced_debug_options(adv_opts):
    """
    check and parse the advanced debug options dictionary for necessary keys and values.
      - dry_run
      - log_file_path
    """
    # 1) dry_run
    dry_run_val = adv_opts.get("dry_run", False)
    if isinstance(dry_run_val, str):
        dry_run_val = dry_run_val.lower() in ["true", "1", "yes"]
    adv_opts["dry_run"] = dry_run_val

    # 2) log_file_path
    log_file_path = adv_opts.get("log_file_path", "NONE").strip()
    adv_opts["log_file_path"] = log_file_path

    logger.debug("Advanced debug options checks passed.")


def handle_log_file(advanced_options, parameters):
    """
    handle the log file path.
    """
    log_file_path = advanced_options.get("log_file_path", "NONE")
    if log_file_path == "NONE" or not log_file_path:
        out_dir = parameters.get("output_dir", ".")
        out_dir = os.path.abspath(out_dir)
        log_dir = os.path.join(out_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "xc.log")
        logger.debug(f"Log file path not specified, using default: {log_file_path}")

    if os.path.isfile(log_file_path):
        os.remove(log_file_path)
        logger.debug(f"Removed existing log file: {log_file_path}")

    with open(log_file_path, "w") as f:
        pass

    logger.info(f"Log file created: {log_file_path}")
    advanced_options["log_file_path"] = log_file_path


def check_future_options(future_options):
    """
    Do some basic checks on future_options if needed.
    """
    source_info_file = future_options.get("source_info_file", "NONE")
    if source_info_file != "NONE":
        if not os.path.isfile(source_info_file):
            raise FileNotFoundError(
                f"source_info_file in [Future_options] does not exist: {source_info_file}"
            )
    logger.debug("Future options checks passed.")


def parse_and_check_ini_file(file_path):
    """
    read the ini file and parse the configuration.
    check the configuration and return the dictionaries.
    """
    logger.info(f"Loading configuration file: {file_path}\n")
    config = load_config_file(file_path)

    # 1) parse the sections
    (
        array_info1,
        array_info2,
        parameters,
        executables,
        device_info,
        adv_proc,
        adv_storage,
        adv_debug,
        future_options,
    ) = parse_sections(config)

    # 2) check the configuration
    check_array_info(array_info1, array_info2)
    check_executables(executables)
    check_parameters(parameters)
    check_device_info(device_info)
    check_advanced_proc_options(adv_proc)
    check_advanced_storage_options(adv_storage)
    check_advanced_debug_options(adv_debug)
    check_future_options(future_options)

    # 3) handle log file
    handle_log_file(adv_debug, parameters)

    logger.info("All configuration checks passed.\n"+'-'*80+'\n')
    return (
        array_info1,
        array_info2,
        parameters,
        executables,
        device_info,
        adv_proc,
        adv_storage,
        adv_debug,
        future_options,
    )


# check the function by running it
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ini_path = os.path.join(current_dir, "template.ini")
    try:
        results = parse_and_check_ini_file(ini_path)
        print("[TEST] Successfully parsed and checked the INI file.")
        print("Returned results:", results)
    except Exception as e:
        print(f"[TEST] Failed to parse/check the INI file: {e}")
