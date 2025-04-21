import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Create a logger
logger = logging.getLogger(__name__)


class FileMatcher:
    """_summary_
    A class-based approach for:
    1. Collecting files from a directory
    2. Matching each file against a regex pattern
    3. Extracting time fields and constructing datetime objects
    """
    def __init__(self, directory: str, regex_pattern: str):
        """
        :param directory: Root directory to walk through.
        :param regex_pattern: Regex pattern with named groups for capturing fields.
        """
        self.directory = directory
        self.regex_pattern = regex_pattern
        self.matched_files: List[Dict] = []

    def get_files(self) -> List[str]:
        """
        Recursively collect all files in `self.directory`.
        """
        file_list = []
        count = 0
        logger.info(f"Searching for files in {self.directory}")
        for root, _, files in os.walk(self.directory):
            for file_name in files:
                file_list.append(os.path.join(root, file_name))
                count += 1
        logger.info(f"Finish. {count} files found in {self.directory}")
        return file_list

    def match_files(self, file_paths: Optional[List[str]] = None, num_threads: int = 1) -> List[Dict]:
        """
        Match a list of files with `self.regex_pattern`, extract fields, and store them in `matched_files`.
        If no file_paths is provided, we will call `get_files()` automatically.
        """
        if file_paths is None:
            file_paths = self.get_files()
        logger.info("Start file pattern matching...")

        # Use ThreadPoolExecutor to parallelize the matching process
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self._match_file, file_paths))

        # filter out None results
        all_results = [res for res in results if res]
        logger.info(f"{len(all_results)} files matched.")
        self.matched_files = all_results  # store the matched files
        return all_results
    
    def _match_file(self, file_path: str) -> Dict:
        """
        Match a single file path against the regex_pattern,
        and parse its time fields if present.
        """
        fields = {}
        try:
            match = re.match(self.regex_pattern, file_path)
            if match:
                fields = match.groupdict()
                # parse time fields if they exist
                fields["time"] = self._gen_time_from_fields(fields)
                # store path
                fields["path"] = file_path
        except Exception as e:
            logger.error(f"An error occurred while processing the file {file_path}: {e}")
        return fields
    
    def _gen_time_from_fields(self, fields: Dict[str, str]) -> Optional[datetime]:
        """
        Construct a datetime object from year/month/day/jday/hour/minute fields in the dict.
        If invalid or incomplete, return None (or raise an error).
        """
        # extract fields
        year = fields.get("year")
        month = fields.get("month")
        day = fields.get("day")
        jday = fields.get("jday")
        hour = fields.get("hour", "0")
        minute = fields.get("minute", "0")

        # convert year to int
        if year is not None:
            if len(year) == 2:
                year = 2000 + int(year)
            else:
                year = int(year)

        # convert other fields to int
        jday = int(jday) if jday else None
        month = int(month) if month else None
        day = int(day) if day else None
        hour = int(hour)
        minute = int(minute)

        if year and jday:
            # calculate time from year and jday
            try:
                return datetime(year, 1, 1) + timedelta(days=jday - 1, hours=hour, minutes=minute)
            except ValueError as e:
                logger.error(f"Invalid jday or date fields: {fields}, error: {e}")
                return None
        elif year and month and day:
            # calculate time from year, month, and day
            try:
                return datetime(year, month, day, hour, minute)
            except ValueError as e:
                logger.error(f"Invalid date fields: {fields}, error: {e}")
                return None
        else:
            logger.error(f"Insufficient time fields: {fields}")
            return None
