import os
import glob
import shutil
import logging
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

logger = logging.getLogger(__name__)

def concatenate_ncf(output_dir: str, cpu_count: int) -> None:
    """
    Traverse subdirectories under ncf/<gpu_x>, collect .bigsac files with the same name, 
    and merge them in binary mode. Merged files will be placed in a temporary directory 
    (temp_ncf), then the original ncf folder is removed and replaced by temp_ncf. 
    Finally, each queue_x directory will be deleted.

    :param output_dir: The base directory containing the 'ncf' folder.
    :param cpu_count:  The number of threads to use for parallel merging.
    """
    ncf_dir = os.path.join(output_dir, "ncf")
    temp_ncf_dir = os.path.join(output_dir, "temp_ncf")

    # Remove existing temp_ncf directory if it exists, then recreate it
    if os.path.exists(temp_ncf_dir):
        shutil.rmtree(temp_ncf_dir)
    os.makedirs(temp_ncf_dir, exist_ok=True)

    # Collect all queue_x subdirectories under ncf
    gpu_subdirs = []
    for sub in os.listdir(ncf_dir):
        subpath = os.path.join(ncf_dir, sub)
        if os.path.isdir(subpath) and sub.startswith("queue_"):
            gpu_subdirs.append(subpath)

    if not gpu_subdirs:
        logger.info("No 'queue_x' subdirectories found under 'ncf'. Skipping aggregation.\n")
        return

    # Map each filename to a list of file paths that share this name
    ncf_file_map = defaultdict(list)

    logger.info(f"Collecting .bigsac files from {len(gpu_subdirs)} GPU subdirectories.\n")
    for gpu_dir in gpu_subdirs:
        sac_files = glob.glob(os.path.join(gpu_dir, "*.bigsac"))
        # In case the above search might fail or be repeated, you could adjust this if needed
        if not sac_files:
            sac_files = glob.glob(os.path.join(gpu_dir, "*.bigsac"))

        for sfile in sac_files:
            fname = os.path.basename(sfile)
            # Ensure the .bigsac extension is standardized
            if fname.endswith(".bigsac"):
                fname = fname[:-7] + ".bigsac"
            ncf_file_map[fname].append(sfile)

    if not ncf_file_map:
        logger.info("No .bigsac files found in any GPU subdirectory. Skipping aggregation.\n")
        return

    total_fnames = len(ncf_file_map)
    logger.info(f"Found {total_fnames} unique .bigsac files to merge.\n")

    def merge_one_fname(fname, paths, delete):
        """
        Merge multiple .bigsac files into a single file in binary mode.
        Optionally delete the source files after merging.
        """
        target_path = os.path.join(temp_ncf_dir, fname)
        with open(target_path, "ab") as out_f:
            for sfile in paths:
                with open(sfile, "rb") as in_f:
                    out_f.write(in_f.read())
                if delete:
                    os.remove(sfile)

    # Use ThreadPoolExecutor to merge files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        for fname, paths in ncf_file_map.items():
            futures.append(executor.submit(merge_one_fname, fname, paths, delete=True))

        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_fnames,
            desc="[Concatenating NCF]",
            unit="file"
        ):
            pass

    # Delete the queue_x directories after merging
    for gpu_dir in gpu_subdirs:
        logger.debug(f"Deleting directory: {gpu_dir}.\n")
        shutil.rmtree(gpu_dir, ignore_errors=True)
        logger.debug(f"Successfully deleted {gpu_dir}.\n")

    # Replace the old ncf directory with the merged results
    if os.path.exists(ncf_dir):
        shutil.rmtree(ncf_dir, ignore_errors=True)
    print("\n")
    logger.info(f"Removing redundant files\n")
    os.rename(temp_ncf_dir, ncf_dir)