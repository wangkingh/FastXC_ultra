#!/usr/bin/env python3
import struct, os, datetime

VERBOSE = False
SAC_HEADER_SIZE = 632

# ---------------- SAC 头偏移（小端） ----------------
NPTS_OFFSET = 79 * 4
IFTYPE_OFFSET = 70 * 4 + 15 * 4  # int 区第 16 个
OFFSET_USER4 = 176

INT_BASE = 70 * 4  # int 区起始 (70 floats = 280 B)
OFFSET_NZYEAR = INT_BASE + 0 * 4
OFFSET_NZJDAY = INT_BASE + 1 * 4
OFFSET_NZHOUR = INT_BASE + 2 * 4
OFFSET_NZMIN = INT_BASE + 3 * 4
OFFSET_NZSEC = INT_BASE + 4 * 4

# ---------- 新增：字符区偏移 ----------
CHAR_BASE = (70 + 40) * 4  # 280 + 160 = 440 B
KSTNM_OFFSET_C = CHAR_BASE  # 8 B
KEVNM_OFFSET_C = CHAR_BASE + 8  # 16 B (= 两块 8 B)

IXY = 4  # 正确的 IXY 枚举
DOUBLE_TYPES = {2, 3, 4}  # 需要 ×2 的 iftype
# ---------------------------------------------------


def read_sac_header(fd):
    """读取 632 B 头并返回 (bytes,npts,iftype,time_str)；不足则返 None"""
    buf = fd.read(SAC_HEADER_SIZE)
    if len(buf) < SAC_HEADER_SIZE:
        return None

    uf = lambda off: struct.unpack_from("<f", buf, off)[0]
    ui = lambda off: struct.unpack_from("<i", buf, off)[0]

    npts = ui(NPTS_OFFSET)
    iftype = ui(IFTYPE_OFFSET)

    # -------- 时间字段 --------
    nzyear = ui(OFFSET_NZYEAR)
    nzjday = ui(OFFSET_NZJDAY)
    nzhour = ui(OFFSET_NZHOUR)
    nzmin = ui(OFFSET_NZMIN)
    nzsec = ui(OFFSET_NZSEC)
    user4 = max(uf(OFFSET_USER4), 0)  # 负值时归零

    # 默认时间串
    time_str = f"{nzyear}.{nzjday:03d}.{nzhour:02d}{nzmin:02d}"

    # 校正中心时刻 → 起始时刻
    if nzhour != -12345 and nzmin != -12345:
        center = datetime.datetime(nzyear, 1, 1) + datetime.timedelta(
            days=nzjday - 1, hours=nzhour, minutes=nzmin, seconds=nzsec
        )
        start = center + datetime.timedelta(seconds=user4)
        jday = (start - datetime.datetime(start.year, 1, 1)).days + 1
        time_str = f"{start.year}.{jday:03d}.{start.hour:02d}{start.minute:02d}"

    return buf, npts, iftype, time_str


def extract_segments(bigfile, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 从文件名取出 kstnm 与 kcmpnm
    _, kstnm, kcmpnm, _ = os.path.basename(bigfile).split(".", 3)

    # ------- 预处理：拆分 kstnm -------
    k_parts = kstnm.split("-", 1)
    k_part1 = k_parts[0][:8].ljust(8)  # kstnm 字段，8 B
    k_part2 = (k_parts[1] if len(k_parts) > 1 else "").ljust(16)[:16]  # kevnm，16 B
    k_part1_b = k_part1.encode("ascii")
    k_part2_b = k_part2.encode("ascii")
    # ----------------------------------

    with open(bigfile, "rb") as fd:
        seg = 0
        while True:
            hdr = read_sac_header(fd)
            if hdr is None:
                break

            head_bytes, npts, iftype, time_str = hdr

            # 把 header 变成可写 bytearray，再写入 kstnm/kevnm
            head = bytearray(head_bytes)
            head[KSTNM_OFFSET_C : KSTNM_OFFSET_C + 8] = k_part2_b
            head[KEVNM_OFFSET_C : KEVNM_OFFSET_C + 16] = k_part1_b
            head_bytes = bytes(head)  # 重新转回 bytes 以写文件

            # ------- 计算数据大小 -------
            samples = npts * (2 if iftype in DOUBLE_TYPES else 1)
            data_bytes = samples * 4
            data = fd.read(data_bytes)
            if len(data) < data_bytes:
                print("Incomplete data block.")
                break
            # ---------------------------

            out_path = os.path.join(
                out_dir,
                f"{kstnm}.{time_str if '-12345' not in time_str else f'seg{seg:04d}'}.{kcmpnm}.sac",
            )
            with open(out_path, "wb") as out_fd:
                out_fd.write(head_bytes)
                out_fd.write(data)

            if VERBOSE:
                print(f"{seg:04d}  {os.path.basename(out_path)}  bytes={data_bytes}")

            seg += 1

    print(f"Extraction {bigfile} finished: {seg} segments.")


# ---------------- 入口 ----------------
if __name__ == "__main__":
    extract_segments(
        "/mnt/g/test_fastxc_output/ncf/VV-VV.AAKH-ABNH.U-U.bigsac", "./test3"
    )
