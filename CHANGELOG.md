## 2025-05-02
- 重大更新：重构核心代码并全面改写 CHANGELOG。
- Major release: refactored core codebase and fully rewrote the CHANGELOG.


## Changelog (2025-05-03)

- **Fix queue dead-lock** – `dispatcher` 现在只在真正转移任务或放回任务后成对调用 `task_done()`，`join()` 不再卡死。  
- **Round-Robin dispatch** – 用 `collections.deque.rotate()` 均衡地把任务分配到各 `(device, worker)` 子队列，避免偏向首个 GPU。  
- **Global unique -Q id** – 为每个 `(device_type, dev_id, worker_id)` 生成全局递增 `gid`，拼进命令行 `-Q {gid}`，解决不同 GPU 上 worker-0 冲突。  
