from typing import Dict, List, Optional
from collections import OrderedDict
from .pattern_utils import FieldRegistry, DEFAULT_BASE_FIELDS, check_pattern
from .file_matcher import FileMatcher
from .file_filter import FileFilter
from .file_organizer import group_by_labels, organize_by_labels
import logging

logger = logging.getLogger(__name__)


class SeisArray:
    """
    SeisArray class is designed for organizing the noise data into a virtual array.
    """

    def __init__(
        self,
        array_dir: str,
        pattern: str,
        custom_fields: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ):
        """
        :param array_dir: root directory of the array
        :param pattern:   pattern (e.g. "{home}/{YYYY}/{station}_{component}.sac")
        :param custom_fields: allow user to define custom fields, e.g.
                              {"shot": r"\d+", "line": r"[A-Z0-9]+"}
        :param overwrite: when adding custom fields, overwrite the existing fields
        """
        base_copy = OrderedDict(DEFAULT_BASE_FIELDS)
        self.registry = FieldRegistry(base_copy)

        if custom_fields:
            # add custom fields to the registry
            for field_name, regex_str in custom_fields.items():
                self.registry.add_field(field_name, regex_str, overwrite=overwrite)

        self.array_dir = array_dir
        self.pattern = check_pattern(array_dir, pattern, self.registry)
        self.files = None
        self.filtered_files = None
        self.pattern_filter = None
        self.files_group = None
        self.virtual_array = None

    def match(self, threads: int = 1):
        """
        Matching files in array_dir using FileMatcher according to self.pattern.
        The matched info is stored in self.files.
        """
        matcher = FileMatcher(directory=self.array_dir, regex_pattern=self.pattern)
        self.files = matcher.match_files(num_threads=threads)

    def filter(
        self,
        criteria: Optional[Dict[str, List]] = None,
        threads: int = 1,
        verbose: bool = False,
    ):
        """
        Apply the file filter (new class-based logic) to the directory,
        and store the matched files.

        :param criteria:
            e.g. : {
                "type": {
                    "type": "list",
                    "data_type": "str"/"datetime"/"float"/"int"/... (optional),
                    "value": ["image", "video"]
                },
                "time": {
                    "type": "range",
                    "data_type": "datetime",      # ensure 'time' in file_info is datetime
                    "value": [
                        "2023-01-01 00:00:00",
                        "2023-01-02 00:00:00",
                        "2023-01-02 12:00:00",
                        "2023-01-03 20:00:00"
                    ]
                },
                "size": {
                    "type": "range",
                    "data_type": "int",           # or "float" if needed
                    "value": [100, 2000]
                }
            }
        :param threads: the number of pool
        """
        if self.files is None:
            logger.warning("Please match the files first.")
            return None

        file_filter = FileFilter(criteria=criteria, num_threads=threads)
        if verbose:
            file_filter.show_criteria()
        self.filtered_files = file_filter.filter_files(self.files)

    def group(self, labels: list, sort_labels: list = None, filtered=True):
        """
        re-organize the array files according to the order
        """
        if filtered:
            files = self.filtered_files
        else:
            files = self.files

        if files is None:
            if filtered:
                logger.error("Please filter the files first.")
            else:
                logger.error("Please match the files first.")
            return None

        files_group = group_by_labels(files, labels, sort_labels)
        self.files_group = files_group.to_dict(orient="index")

    def organize(self, label_order: list, output_type="dict", filtered=True):
        """
        re-organize the array files according to the order
        """
        if filtered:
            files = self.filtered_files
        else:
            files = self.files

        if files is None:
            if filtered:
                logger.error("Please filter the files first.")
            else:
                logger.error("Please match the files first.")
            return None

        if output_type not in ["path", "dict"]:
            logger.error("[Error] flag should be 'path' or 'dict'.")
            output_type = "dict"
        self.virtual_array = organize_by_labels(files, label_order, output_type)

    def get_stations(self, filtered: bool = True) -> list:
        station_set = []
        if filtered:
            files = self.filtered_files
        else:
            files = self.files
        for file_info in files:
            station_set.append(file_info["station"])
    
    def get_times(self, filtered: bool = True) -> list:
        station_set = []
        if filtered:
            files = self.filtered_files
        else:
            files = self.files
        for file_info in files:
            station_set.append(file_info["time"])
