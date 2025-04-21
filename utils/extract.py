import struct
import os

# SAC header size (bytes), 通常为 632 字节。如果不同，请根据实际情况修改
SAC_HEADER_SIZE = 632

# 根据SAC规范，header部分包含固定数量的float,int和char字段，此处假设:
# npts在头的某个固定位置（具体偏移需根据SAC格式手册或已有代码）
# iftype在另一个固定位置，同理。
# 以下offset只是示例，请根据SAC手册或C代码中的定义修改
NPTS_OFFSET = 79 * 4  # SAC header中的npts是第80个float字段(从0开始计数时是79*4字节处)
IFTYPE_OFFSET = 86 * 4  # SAC中iftype通常是第87个float字段（实际要查看官方文档或源码）
# 注意：实际SAC头结构中iftype是int类型字段，需要根据官方定义查看其偏移。


def read_sac_header(fd):
    SAC_HEADER_SIZE = 632
    head_bytes = fd.read(SAC_HEADER_SIZE)
    if len(head_bytes) < SAC_HEADER_SIZE:
        return None

    NPTS_OFFSET = 79 * 4  # 示例，和你原有的一致
    IFTYPE_OFFSET = 86 * 4  # 示例，和你原有的一致

    # 这里假设小端 (little-endian)，如果是大端需要改成 ">i"
    npts = struct.unpack_from("<i", head_bytes, NPTS_OFFSET)[0]
    iftype = struct.unpack_from("<i", head_bytes, IFTYPE_OFFSET)[0]

    # 新增读取 nzyear, nzjday, nzhour, nzmin
    NZYEAR_OFFSET = 280
    NZJDAY_OFFSET = 284
    NZHOUR_OFFSET = 288
    NZMIN_OFFSET = 292

    nzyear = struct.unpack_from("<i", head_bytes, NZYEAR_OFFSET)[0]
    nzjday = struct.unpack_from("<i", head_bytes, NZJDAY_OFFSET)[0]
    nzhour = struct.unpack_from("<i", head_bytes, NZHOUR_OFFSET)[0]
    nzmin = struct.unpack_from("<i", head_bytes, NZMIN_OFFSET)[0]

    # 拼出字符串: YYYY.JJJ.HHMI
    # 注意JDAY要3位数(如 Day 1 -> 001, Day 45 -> 045)
    # 注意HOUR, MIN要2位数(01, 09, 10等)
    time_str = f"{nzyear}.{nzjday:03d}.{nzhour:02d}{nzmin:02d}"

    # 将这几个值一并返回 (你也可以只返回 time_str)
    return head_bytes, npts, iftype, time_str


def extract_segments(bigfile, output_dir):
    """
    从大文件中提取多个SAC段。每个段包含一个SAC头和对应的数据。
    将其写出为独立的SAC文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(bigfile, "rb") as fd:
        segment_index = 0
        fname = os.path.basename(bigfile)
        kstnm, kcmpnm, _ = fname.split(".")
        while True:
            # 尝试读取下一个头
            pos = fd.tell()
            header_info = read_sac_header(fd)
            if header_info is None:
                # 读不到更多数据，结束
                break

            head_bytes, npts, iftype, time_str = header_info

            # 根据iftype和npts确定数据大小
            # 若iftype==IXY，需要2*npts个float数据，否则就是npts个float数据
            # 在C中定义IXY常量的值是多少，需要保持一致，这里假设IXY=1做示例
            IXY = 1  # 实际值请根据SAC定义修改
            data_count = npts
            if iftype == IXY:
                data_count = npts * 2

            data_size = data_count * 4  # float数据，4字节一个

            # 读取数据
            data_bytes = fd.read(data_size)
            if len(data_bytes) < data_size:
                # 数据不完整，结束或报错
                print("Incomplete data at the end of file.")
                break

            # 将这一段写出到单独的文件
            out_name = f"{kstnm}.{time_str}.{kcmpnm}.sac"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "wb") as out_fd:
                out_fd.write(head_bytes)
                out_fd.write(data_bytes)

            print(f"Extracted segment {segment_index} to {out_path}")
            segment_index += 1

    print("Extraction finished.")


if __name__ == "__main__":

    # extract_segments("/mnt/c/Users/admin/Desktop/FastXC_test/jls_cc/ncf/233-334.Z-Z.sac", "./test")

    extract_segments("/mnt/f/hinet_cc/ncf/AAKH-ABNH.U-U.bigsac", "./test3")
