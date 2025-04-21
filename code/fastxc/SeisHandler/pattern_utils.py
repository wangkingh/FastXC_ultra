import os
import re
import logging
from typing import Dict
from collections import OrderedDict, Counter
import logging

logger = logging.getLogger(__name__)


class FieldRegistry:
    def __init__(self, base_fields: Dict[str, str]):
        """_summary_

        Args:
            base_fields (Dict[str, str]):
            as "YYYY": r"(?P<year>\d{4})",  # 4 digits for year
        """
        if base_fields is None:
            base_fields = {}  # empty dictionary
        # copy the base fields
        self._fields = OrderedDict(base_fields)

    def add_field(self, field_name: str, regex_str: str, overwrite: bool = False):
        """_summary_

        Args:
            field_name (str):
            regex_str (str):
            overwrite (bool, optional):
        """
        if field_name in self._fields and not overwrite:
            logger.warning(
                "Field %s already exists! Use overwrite=True to overwrite.", field_name
            )
            return

        named_group = f"(?P<{field_name}>{regex_str})"

        # test the regex
        try:
            re.compile(named_group)
        except re.error as e:
            logger.error("Invalid regex pattern: %s", e)
            raise ValueError(f"Invalid regex pattern: {e}")

        self._fields[field_name] = named_group

    def remove_field(self, field_name: str):
        """_summary_

        Args:
            field_name (str):
        """
        if field_name in self._fields:
            del self._fields[field_name]
        else:
            logger.warning("Field %s not found.", field_name)

    def get_fields(self):
        return self._fields

    def validate_pattern_fields(self, pattern: str):
        """_summary_

        Args:
            pattern (str):
        """
        pattern_fields = set(re.findall(r"\{(\w+)}", pattern))
        valid_fields = set(self._fields.keys())
        if not pattern_fields.issubset(valid_fields):
            invalid_fields = pattern_fields - valid_fields
            logger.error("Pattern contains invalid fields: %s", invalid_fields)
            raise ValueError(f"pattern contains invalid fields: {invalid_fields}")

    def build_regex_pattern(self, pattern: str):
        """_summary_

        Args:
            pattern (str):
        """
        # Replace field names with corresponding regex patterns
        for field_name, regex in self._fields.items():
            pattern = pattern.replace("{" + field_name + "}", regex)
        # Escape special characters and compile the final regex pattern
        pattern = pattern.replace(".", r"\.")
        pattern = pattern.replace("_", r"\_")
        pattern = pattern.replace("/", r"\/")
        # Replace '?' (any character wildcard) with regex for any characters except for special characters
        pattern = pattern.replace("{?}", "[^. _/]*")
        # Replace '*'(any character wildcard) with regex for any characters
        pattern = pattern.replace("{*}", ".*")
        return r"{}".format(pattern)


# making this a global variable
DEFAULT_BASE_FIELDS = OrderedDict(
    {
        "YYYY": r"(?P<year>\d{4})",  # 4 digits for year
        "YY": r"(?P<year>\d{2})",  # 2 digits for year
        "MM": r"(?P<month>\d{2})",  # 2 digits for month
        "DD": r"(?P<day>\d{2})",  # 2 digits for day
        "JJJ": r"(?P<jday>\d{3})",  # 3 digits for day of year
        "HH": r"(?P<hour>\d{2})",  # 2 digits for hour
        "MI": r"(?P<minute>\d{2})",  # 2 digits for minute
        "home": r"(?P<home>\w+)",  # for home directory
        "network": r"(?P<network>\w+)",  # for network code
        "event": r"(?P<event>\w+)",  # for network code
        "station": r"(?P<station>\w+)",  # for station name
        "component": r"(?P<component>\w+)",  # for component name
        "sampleF": r"(?P<sampleF>\w+)",  # for sampling frequency
        "quality": r"(?P<quality>\w+)",  # for quality indicator
        "locid": r"(?P<locid>\w+)",  # for location ID
        "suffix": r"(?P<suffix>\w+)",  # for file extension
        "label0": r"(?P<label0>\w+)",  # for file label0
        "label1": r"(?P<label1>\w+)",  # for file label1
        "label2": r"(?P<label2>\w+)",  # for file label2
        "label3": r"(?P<label3>\w+)",  # for file label3
        "label4": r"(?P<label4>\w+)",  # for file label4
        "label5": r"(?P<label5>\w+)",  # for file label5
        "label6": r"(?P<label6>\w+)",  # for file label6
        "label7": r"(?P<label7>\w+)",  # for file label7
        "label8": r"(?P<label8>\w+)",  # for file label8
        "label9": r"(?P<label9>\w+)",  # for file label9
    }
)


def check_pattern(array_dir: str, pattern: str, registry: FieldRegistry) -> str:
    """
    Check if pattern is a valid string and return a dictionary with
    """

    if not isinstance(pattern, str):
        logger.error("Pattern must be a string, but got %s", type(pattern))
        raise TypeError("pattern must be a string")

    # check if all fields in the pattern are valid
    registry.validate_pattern_fields(pattern)

    # avoid duplicate fields
    pattern_fields_list = re.findall(r"\{(\w+)}", pattern)
    field_counts = Counter(pattern_fields_list)
    duplicate_fields = [field for field, count in field_counts.items() if count > 1]
    if duplicate_fields:
        logger.error("Pattern contains duplicate fields: %s", duplicate_fields)
        raise ValueError(f"pattern contains duplicate fields: {duplicate_fields}")

    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")

    # check if necessary fields are in the pattern
    necessary_fields = ["{home}", "{component}", "{station}"]
    for f in necessary_fields:
        if f not in pattern:
            logger.error("Pattern must contain %s", f)
            raise ValueError(f"pattern must contain {f}")

    # check if one of the date fields is in the pattern
    date_fields0 = ["{YYYY}", "{MM}", "{DD}"]
    date_fields1 = ["{YYYY}", "{JJJ}"]
    date_fields2 = ["{YY}", "{MM}", "{DD}"]
    date_fields3 = ["{YY}", "{JJJ}"]
    if not any(
        field in pattern
        for field in date_fields0 + date_fields1 + date_fields2 + date_fields3
    ):
        logger.error("Pattern must contain one set of date fields.")
        raise ValueError("pattern must contain one set of date fields")

    # check is sac_dir is a dir, else warning
    if not os.path.isdir(array_dir):
        logger.error("%s is not a directory", array_dir)

    # Apply the sac_dir to the pattern
    pattern = pattern.replace("{home}", os.path.normpath(array_dir))

    # Create the regex pattern
    regex_pattern = registry.build_regex_pattern(pattern)
    return regex_pattern
