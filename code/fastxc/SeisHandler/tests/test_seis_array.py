import unittest
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional

from SeisHandler.seis_array import SeisArray

class TestSeisArray(unittest.TestCase):

    def setUp(self):
        """
        每个测试方法前都会执行：
        1) 创建一个临时目录，
        2) 在部分用例里写入测试文件，
        3) 初始化 SeisArray 时可根据需要传不同的 pattern。
        """
        self.tmpdir = tempfile.TemporaryDirectory()
        self.test_dir = self.tmpdir.name

    def tearDown(self):
        """测试结束时自动清理临时目录。"""
        self.tmpdir.cleanup()

    # ----------------------------------------------------------------------
    # 1) 最基本的匹配 -> 过滤 -> 分组 -> 组织 流程（你已有的test_basic_flow）
    # ----------------------------------------------------------------------
    def test_basic_flow(self):
        """
        测试最基本的：匹配 -> 过滤 -> 分组 -> 组织
        使用 pattern: {home}/{YYYY}/{station}_{component}.sac
        """
        # 1) 创建测试目录并文件
        test_2023_dir = os.path.join(self.test_dir, "2023")
        os.makedirs(test_2023_dir, exist_ok=True)

        file_path_1 = os.path.join(test_2023_dir, "ABC_BHZ.sac")
        file_path_2 = os.path.join(test_2023_dir, "DEF_LHZ.sac")
        file_path_3 = os.path.join(test_2023_dir, "XYZ_BHZ.sac")
        for fp in [file_path_1, file_path_2, file_path_3]:
            with open(fp, "w") as f:
                f.write("Test data")

        # 2) 初始化 SeisArray
        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)

        # 3) match
        sa.match(threads=1)
        self.assertIsNotNone(sa.files)
        self.assertEqual(len(sa.files), 3)

        stations = sorted([f["station"] for f in sa.files])
        components = sorted([f["component"] for f in sa.files])
        self.assertEqual(stations, ["ABC", "DEF", "XYZ"])
        self.assertEqual(components, ["BHZ", "BHZ", "LHZ"])

        # 4) filter
        criteria = {
            "station": {
                "type": "list",
                "data_type": "str",
                "value": ["ABC", "XYZ"]
            }
        }
        sa.filter(criteria=criteria, threads=1)
        self.assertIsNotNone(sa.filtered_files)
        self.assertEqual(len(sa.filtered_files), 2)

        # 5) group
        sa.group(labels=["station"], sort_labels=["station"], filtered=True)
        self.assertIn("ABC", sa.files_group)
        self.assertIn("XYZ", sa.files_group)
        self.assertEqual(len(sa.files_group), 2)

        # 6) organize
        sa.organize(label_order=["station"], output_type="dict", filtered=True)
        self.assertIsNotNone(sa.virtual_array)

        top_keys = sorted(list(sa.virtual_array.keys()))
        self.assertEqual(top_keys, ["ABC", "XYZ"])

        abc_entries = sa.virtual_array["ABC"]
        self.assertTrue(isinstance(abc_entries, list))
        self.assertEqual(len(abc_entries), 1)
        self.assertEqual(abc_entries[0]["station"], "ABC")

    # ----------------------------------------------------------------------
    # 2) 测试日期字段过滤
    # ----------------------------------------------------------------------
    def test_time_filter(self):
        """
        在 pattern 中使用 {home}/{YYYY}/{MM}/{DD}/{station}_{component}.sac
        让 FileMatcher 能解析出 time 字段(精确到天)。
        然后对 time 做 range 过滤。
        """
        # 创建测试目录，并根据日期字段来做子目录
        abc_day1_dir = os.path.join(self.test_dir, "2023", "01", "01")
        abc_day2_dir = os.path.join(self.test_dir, "2023", "01", "02")
        xyz_day1_dir = os.path.join(self.test_dir, "2023", "01", "01")

        os.makedirs(abc_day1_dir, exist_ok=True)
        os.makedirs(abc_day2_dir, exist_ok=True)
        os.makedirs(xyz_day1_dir, exist_ok=True)

        # 创建若干文件
        #  2023/01/01/ABC_BHZ.sac -> time=2023-01-01
        #  2023/01/02/ABC_BHZ.sac -> time=2023-01-02
        #  2023/01/01/XYZ_BHZ.sac -> time=2023-01-01
        file1 = os.path.join(abc_day1_dir, "ABC_BHZ.sac")
        file2 = os.path.join(abc_day2_dir, "ABC_BHZ.sac")
        file3 = os.path.join(xyz_day1_dir, "XYZ_BHZ.sac")
        for fp in [file1, file2, file3]:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as f:
                f.write("Test data")

        # 构造包含日期占位符的 pattern
        pattern = "{home}/{YYYY}/{MM}/{DD}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)
        sa.match()

        # 确认匹配到 3 个文件
        self.assertIsNotNone(sa.files)
        self.assertEqual(len(sa.files), 3)

        # 检查 time 字段是不是 datetime 对象
        # ABC_BHZ / 2023/01/01
        times_sorted = sorted([f["time"] for f in sa.files])
        # times_sorted 应该是 datetime(2023,1,1), datetime(2023,1,1), datetime(2023,1,2)
        self.assertTrue(all(isinstance(t, datetime) for t in times_sorted))

        # 定义过滤: 只保留 time >= 2023-01-02
        start = datetime(2023,1,2)
        end   = datetime(2023,1,2,23,59,59)
        criteria = {
            "time": {
                "type": "range",
                "data_type": "datetime",
                "value": [start, end]
            }
        }
        sa.filter(criteria=criteria)
        self.assertIsNotNone(sa.filtered_files)
        # 理论上只剩下 1 个文件(2023/01/02/ABC_BHZ.sac)
        self.assertEqual(len(sa.filtered_files), 1)
        self.assertEqual(sa.filtered_files[0]["station"], "ABC")

    # ----------------------------------------------------------------------
    # 3) 测试自定义字段
    # ----------------------------------------------------------------------
    def test_custom_fields(self):
        """
        在 pattern 中添加 {shot} 这样一个自定义字段
        并查看它在 matched_files 里是否正确解析
        """
        test2023_dir = os.path.join(self.test_dir, "2023")
        os.makedirs(test2023_dir, exist_ok=True)

        # 文件示例: /tmpdir/2023/ABC_123_BHZ.sac
        # 这里把 {shot} 定义成 r"\d+"
        file1 = os.path.join(test2023_dir, "ABC_123_BHZ.sac")
        file2 = os.path.join(test2023_dir, "ABC_456_BHZ.sac")
        with open(file1, "w") as f:
            f.write("data1")
        with open(file2, "w") as f:
            f.write("data2")

        # 给 SeisArray 传入 custom_fields
        # pattern 里加上 {shot}，比如: {home}/{YYYY}/{station}_{shot}_{component}.sac
        custom_fields = {"shot": r"\d+"}
        pattern = "{home}/{YYYY}/{station}_{shot}_{component}.sac"

        sa = SeisArray(
            array_dir=self.test_dir,
            pattern=pattern,
            custom_fields=custom_fields,
            overwrite=False
        )
        sa.match()
        self.assertIsNotNone(sa.files)
        self.assertEqual(len(sa.files), 2)

        # 检查解析
        for fdict in sa.files:
            self.assertIn("shot", fdict, "应该包含自定义字段 shot")
            self.assertTrue(fdict["shot"] in ["123","456"])

    # ----------------------------------------------------------------------
    # 4) 异常场景：空目录、无文件
    # ----------------------------------------------------------------------
    def test_empty_directory(self):
        """
        如果目录里没有任何文件，match() 后 sa.files 应该是空列表或 None。
        """
        # 注意: setUp 默认已经创建了一个临时目录，但我们没放任何文件
        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)
        sa.match()
        # match() 里若找不到文件，则 sa.files 可能是 [] or None, 视具体实现
        self.assertTrue(sa.files == [] or sa.files is None, "空目录应匹配不到任何文件")

    def test_invalid_pattern(self):
        """
        如果传入一个不合法的 pattern(比如缺少 station/component/home等关键字段),
        可能会抛 ValueError。
        """
        # 缺少 {station} => 这个pattern是不合法的
        invalid_pattern = "{home}/{YYYY}/some_{component}.sac"

        with self.assertRaises(ValueError):
            SeisArray(array_dir=self.test_dir, pattern=invalid_pattern)

    def test_filter_without_match(self):
        """
        不先 match 就直接 filter，看看会不会 logger.warning 或者返回 None
        """
        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)

        # 直接 filter
        with patch('logging.Logger.warning') as mock_warn:
            result = sa.filter(criteria={"station":{"type":"list","value":["ABC"]}})
            mock_warn.assert_called_once()
            self.assertIsNone(result, "未匹配文件就 filter, 理应返回 None")

    # ----------------------------------------------------------------------
    # 5) 多线程
    # ----------------------------------------------------------------------
    def test_multi_thread_match(self):
        """
        测试多线程下 match() 是否可行
        """
        # 创建 10 个文件
        test_2023_dir = os.path.join(self.test_dir, "2023")
        os.makedirs(test_2023_dir, exist_ok=True)
        for i in range(10):
            fname = f"STA{i:02d}_BHZ.sac"
            fpath = os.path.join(test_2023_dir, fname)
            with open(fpath, "w") as f:
                f.write("multithread test")

        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)
        sa.match(threads=4)
        self.assertIsNotNone(sa.files)
        self.assertEqual(len(sa.files), 10, "应匹配到10个文件, multi-thread不影响结果")

    def test_multi_thread_filter(self):
        """
        多线程过滤下, 测试结果是否一致
        """
        # 先创建一些文件
        test_2023_dir = os.path.join(self.test_dir, "2023")
        os.makedirs(test_2023_dir, exist_ok=True)
        for i in range(5):
            fname = f"STA{i:02d}_BHZ.sac"
            fpath = os.path.join(test_2023_dir, fname)
            with open(fpath, "w") as f:
                f.write("data")

        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        sa = SeisArray(array_dir=self.test_dir, pattern=pattern)
        sa.match(threads=1)
        self.assertEqual(len(sa.files), 5)

        # 过滤: station in [STA00, STA01, STA02]
        criteria = {
            "station": {
                "type": "list",
                "data_type": "str",
                "value": ["STA00", "STA01", "STA02"]
            }
        }
        sa.filter(criteria=criteria, threads=4)
        self.assertEqual(len(sa.filtered_files), 3)

if __name__ == '__main__':
    unittest.main()
