# Changelog

## [Unreleased] – 2025-05-07
### Added
- **Build system**
  - Guard `MODE` in top-level Makefile; allowed values are `par` or `seq`.
  - Distinct NVCC flag sets for *Debug* (`-g -O3 -lineinfo`) and *Release* (`-O3 --generate-line-info`).

### Changed
- **Makefiles**
  - Unified formatting, colourised section banners, and recursive goals for all CUDA-C sub-projects.
  - Debug build now keeps line-number mapping while retaining fortify‐source checks.

- **`ncf_pws/cuda.main.cu`**
  - Replaced hard-coded `snprintf(…, 8192, …)` with `snprintf(…, sizeof buf, …)` (three occurrences).
  - Removed unused placeholder variable `src_info_file`.

- **`rotate/arguproc.c`**
  - Initialised `inputFile` and `outputFile` to `NULL`; eliminates *maybe-uninitialised* warnings.

- **`xc/cuda.main.cu`**
  - Dropped unused variable `srcinfo_file`.

- **`config/test_c3.ini`**
  - Updated executable paths to match new `bin/` output directory.

- **`run.py`**
  - Adjusted binary lookup to `../../bin/`.

### Fixed
- Potential buffer-overflow warnings from `snprintf` exceeding 1 KiB static buffer.
- False positives for uninitialised file pointers in *rotate* module.

### Removed
- Redundant `--generate-line-info` when `-G` is in use (NVCC); avoids conflicting-flag warnings.

## 2025-05-02
- 重大更新：重构核心代码并全面改写 CHANGELOG。
- Major release: refactored core codebase and fully rewrote the CHANGELOG.


## Changelog (2025-05-03)

- **Fix queue dead-lock** – `dispatcher` 现在只在真正转移任务或放回任务后成对调用 `task_done()`，`join()` 不再卡死。  
- **Round-Robin dispatch** – 用 `collections.deque.rotate()` 均衡地把任务分配到各 `(device, worker)` 子队列，避免偏向首个 GPU。  
- **Global unique -Q id** – 为每个 `(device_type, dev_id, worker_id)` 生成全局递增 `gid`，拼进命令行 `-Q {gid}`，解决不同 GPU 上 worker-0 冲突。  
