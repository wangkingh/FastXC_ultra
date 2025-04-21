import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Create a logger
logger = logging.getLogger(__name__)


class FileFilter:
    """_summary_
    - Integrate the old filter logic into a class.
    - two type include, "list" and "range"
    """

    def __init__(
        self, criteria: Optional[Dict[str, List]] = None, num_threads: int = 1
    ):
        """_summary_

        :params criteria: dicr, each key is the field name, and the value is a dict, include
        {
            "type": str, "list" or "range"
            "value": [...]
        }
        e.g.
        criteria = {
            "type": {
                "type": "list",
                "data_type": "str",
                "value": ["image", "video"]
            },
            "time": {
                "type": "range",
                "data_type": "datetime",
                "value": [
                    "2023-01-01 00:00:00",
                    "2023-01-02 00:00:00",
                    "2023-01-02 12:00:00",
                    "2023-01-03 20:00:00"
                ]
            },
            "size": {
                "type": "range",
                "data_type": "int",
                "value": [100, 2000]
            }
        }
        """
        # raw criteria
        self.raw_criteria = criteria or {}

        # separate the list and range criteria
        self.list_criteria: Dict[str, List[Any]] = {}
        self.range_criteria: Dict[str, List[tuple]] = {}

        # record the declared data type of each field
        self.type_map: Dict[str, Optional[str]] = {}

        self.num_threads = num_threads

        self._parse_criteria()

    def _parse_criteria(self) -> None:
        """Loop through the raw criteria and dispatch to the correct handler."""
        for field_name, cfg in self.raw_criteria.items():
            # cfg should be like {"type": "list"/"range", "value": ...}
            if not isinstance(cfg, dict) or "type" not in cfg or "value" not in cfg:
                logger.error(
                    f"Field '{field_name}' is invalid. Must contain 'type' and 'value'. Skipped."
                )
                sys.exit(1)

            filter_type = cfg["type"]
            declared_type = cfg.get("data_type")
            values = cfg["value"]

            # store the declared type
            self.type_map[field_name] = declared_type

            if filter_type == "list":
                self._parse_list_criteria(field_name, values)
            elif filter_type == "range":
                self._parse_range_criteria(field_name, values)
            else:
                logger.error(
                    f"Field '{field_name}' has unknown filter type '{filter_type}'. Skipped."
                )

    def _parse_list_criteria(self, field_name: str, values) -> None:
        """Handle the 'list' type criteria."""
        # filtering by list: 'values' itself should be a list of valid items
        if not isinstance(values, list):
            logger.error(
                f"Field '{field_name}' claims 'list' type but 'value' is not a list: {values}"
            )
            return

        # store directly
        self.list_criteria[field_name] = values

    def _parse_range_criteria(self, field_name: str, values) -> None:
        """Handle the 'range' type criteria."""
        if not isinstance(values, list):
            logger.error(
                f"Field '{field_name}' claims 'range' type but 'value' is not a list: {values}"
            )
            return

        # if the number of values is odd, discard the last one
        if len(values) % 2 != 0:
            logger.warning(
                f"Field '{field_name}' has an odd number of range items, discarding the last one."
            )
            values = values[:-1]

        range_pairs = []
        for i in range(0, len(values), 2):
            start_val = values[i]
            end_val = values[i + 1]

            # no more do datetime conversion here, assume the user has done it
            range_pairs.append((start_val, end_val))

        if range_pairs:
            self.range_criteria[field_name] = range_pairs

    # -----------------------------------------------------------------------
    # Checking methods
    # -----------------------------------------------------------------------
    def _check_file_in_list_criteria(self, file_info: Dict) -> bool:
        """Check if file_info meets all 'list' criteria."""
        for field_name, valid_list in self.list_criteria.items():
            if field_name not in file_info:
                return False

            declared_type = self.type_map.get(field_name, None)
            file_value = file_info[field_name]

            # 1) optional type check
            if declared_type is not None:
                if not self._check_type(file_value, declared_type):
                    return False

            # 2) check membership
            if file_value not in valid_list:
                return False
        return True

    def _check_file_in_range_criteria(self, file_info: Dict) -> bool:
        """Check if the file_info meets all 'range' criteria."""
        for field_name, pairs in self.range_criteria.items():
            if field_name not in file_info:
                return False

            declared_type = self.type_map.get(field_name, None)
            file_value = file_info[field_name]
            if declared_type and not self._check_type(file_value, declared_type):
                logger.warning(
                    f"Field '{field_name}' has an invalid type '{type(file_value)}'."
                )
                return False

            in_any_range = False

            for start, end in pairs:
                if start <= file_value <= end:
                    in_any_range = True
                    break

            if not in_any_range:
                return False

        return True

    def _check_type(self, val: Any, declared_type: str) -> bool:
        """
        A helper method to check whether `val` matches the declared_type.
        Return True if it matches, or False otherwise.
        """
        if declared_type == "datetime":
            return isinstance(val, datetime)
        elif declared_type in ("float", "int", "numeric"):
            # as long as it can be cast to float we consider it numeric
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False
        elif declared_type == "str":
            return isinstance(val, str)
        # if no recognized declared_type, we just skip check
        return True

    def _is_valid_file(self, file_info: Dict) -> bool:
        """check if the file_info is valid"""
        # check list
        if not self._check_file_in_list_criteria(file_info):
            return False

        # check range
        if not self._check_file_in_range_criteria(file_info):
            return False

        return True

    def filter_files(self, file_list: List[Dict]) -> List[Dict]:
        """
        filter the file_list based on the criteria
        """
        if not file_list:
            logger.warning("No files provided for filtering.")
            return []

        logger.debug("filtering files...")
        # parallel filtering
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(self._is_valid_file, file_list))

        # filter the files according to the results
        filtered_files = [f for f, valid in zip(file_list, results) if valid]

        logger.info(f"filtering finished, {len(filtered_files)} files passed.")

        return filtered_files

    def show_criteria(self) -> None:
        logger.debug("===== Filter Criteria Summary =====")

        logger.debug("List Criteria:")
        for field_name, values in self.list_criteria.items():
            declared_type = self.type_map.get(field_name, None)
            logger.debug(
                f"  - Field '{field_name}' [Type: {declared_type or 'N/A'}] => {values}"
            )

        logger.debug("Range Criteria:")
        for field_name, pairs in self.range_criteria.items():
            declared_type = self.type_map.get(field_name, None)
            logger.debug(
                f"  - Field '{field_name}' [Type: {declared_type or 'N/A'}] => {pairs}"
            )

        logger.debug("Type Map (field: declared_type):")
        for field_name, declared_type in self.type_map.items():
            logger.debug(f"  - {field_name}: {declared_type}")
