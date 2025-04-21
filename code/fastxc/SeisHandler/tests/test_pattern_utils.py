import unittest
import re
import os
import tempfile
from collections import OrderedDict
from SeisHandler.pattern_utils import FieldRegistry, DEFAULT_BASE_FIELDS, check_pattern


class TestPatternUtils(unittest.TestCase):

    def setUp(self):
        """
        run at the beginning of each test method.
        """
        # initialize a FieldRegistry with DEFAULT_BASE_FIELDS
        self.registry = FieldRegistry(DEFAULT_BASE_FIELDS)

        # create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.array_dir = self.temp_dir.name

    def tearDown(self):
        """
        destroy at the end of each test method.
        """
        self.temp_dir.cleanup()

    def test_field_registry_init(self):
        """
        if FieldRegistry is initialized with DEFAULT_BASE_FIELDS
        """
        fields = self.registry.get_fields()
        self.assertIsInstance(fields, OrderedDict)
        self.assertIn("YYYY", fields)
        self.assertIn("station", fields)
        self.assertIn("component", fields)

    def test_field_registry_add_field(self):
        """
        using add_field to add a new field to the registry
        """
        self.assertNotIn("shot", self.registry.get_fields())
        self.registry.add_field("shot", r"\d+")
        self.assertIn("shot", self.registry.get_fields())
        # make sure the regex is correctly added
        self.assertRegex(self.registry.get_fields()["shot"], r"\(\?P<shot>\\d\+\)")

    def test_field_registry_add_existing_field_no_overwrite(self):
        """
        if add_field is called with an existing field and overwrite=False
        """
        # make sure "YYYY" is already in the registry
        self.assertIn("YYYY", self.registry.get_fields())
        # add a new field with the same name, but different regex
        self.registry.add_field("YYYY", r"\d{8}", overwrite=False)
        # add_field should not overwrite the existing field
        self.assertRegex(self.registry.get_fields()["YYYY"], r"\(\?P<year>\\d\{4\}\)")

    def test_field_registry_add_existing_field_overwrite(self):
        """
        add_field with an existing field and overwrite=TrueS
        """
        # "YYYY" is already in the registry
        self.registry.add_field("YYYY", r"\d{8}", overwrite=True)
        self.assertRegex(self.registry.get_fields()["YYYY"], r"\(\?P<YYYY>\\d\{8\}\)")

    def test_field_registry_add_invalid_regex(self):
        """
        if when add_field is called with an invalid regex
        """
        with self.assertRaises(ValueError):
            self.registry.add_field("invalidRegex", r"[abc")  # missing closing bracket

    def test_field_registry_remove_field(self):
        """
        if remove_field can remove a field from the registry
        """
        self.assertIn("YYYY", self.registry.get_fields())
        self.registry.remove_field("YYYY")
        self.assertNotIn("YYYY", self.registry.get_fields())

    def test_field_registry_validate_pattern_fields(self):
        """
        if validate_pattern_fields can check if a pattern contains all fields in the registry
        """
        # format string with all fields
        pattern_ok = "{YYYY}/{station}_{component}.sac"
        # format string with a non-existent field
        pattern_bad = "{YYYY}/{foo}_{component}.sac"

        # for a valid pattern, should not raise any exceptions
        self.registry.validate_pattern_fields(pattern_ok)

        # for a pattern with a non-existent field, should raise a ValueError
        with self.assertRaises(ValueError):
            self.registry.validate_pattern_fields(pattern_bad)

    def test_field_registry_build_regex_pattern(self):
        """
        test build_regex_pattern with a pattern
        """
        pattern = "{YYYY}/{station}.{component}_{?}/{*}"
        reg_str = self.registry.build_regex_pattern(pattern)

        # make sure all fields are replaced
        self.assertIn("(?P<year>\\d{4})", reg_str)
        # make sure '.' '_' '/' are escaped
        self.assertIn(r"\.", reg_str)
        # make sure '{?}' -> '[^. _/]*'
        self.assertIn("[^. _/]*", reg_str)
        # make sure '{*}' -> '.*'
        self.assertIn(".*", reg_str)

    def test_check_pattern_valid(self):
        """
        test check_pattern with a valid pattern
        """
        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        regex_str = check_pattern(self.array_dir, pattern, self.registry)
        self.assertIsInstance(regex_str, str)
        # regex_str should contain the array_dir
        unescaped = (
            regex_str.replace("\\/", "/")  # 把 \/ 还原成 /
            .replace("\\_", "_")  # 把 \_ 还原成 _
            .replace("\\.", ".")  # 把 \. 还原成 .
        )
        self.assertIn(self.array_dir, unescaped)

    def test_check_pattern_duplicate_fields(self):
        """
        if check_pattern is called with a pattern that contains duplicate fields
        """
        pattern_dup = "{home}/{YYYY}/{YYYY}_{station}.sac"
        with self.assertRaisesRegex(ValueError, "duplicate fields"):
            check_pattern(self.array_dir, pattern_dup, self.registry)

    def test_check_pattern_missing_necessary_fields(self):
        """
        if check_pattern is called with a pattern that is missing necessary fields
        """
        pattern = "{YYYY}/{station}.sac"  # lacks "{home}"
        with self.assertRaisesRegex(ValueError, "pattern must contain {home}"):
            check_pattern(self.array_dir, pattern, self.registry)

    def test_check_pattern_no_date_fields(self):
        """
        when check_pattern is called with a pattern that does not contain any date fields
        """
        pattern_no_date = "{home}/{station}_{component}.sac"
        with self.assertRaisesRegex(
            ValueError, "pattern must contain one set of date fields"
        ):
            check_pattern(self.array_dir, pattern_no_date, self.registry)

    def test_check_pattern_not_a_directory(self):
        """
        test check_pattern with a non-directory path
        """
        # create a file in the temp directory
        not_dir_path = os.path.join(self.array_dir, "myfile.txt")
        with open(not_dir_path, "w") as f:
            f.write("test")

        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        # check_pattern should log a warning
        regex_str = check_pattern(not_dir_path, pattern, self.registry)
        self.assertIn("myfile\\.txt", regex_str)


if __name__ == "__main__":
    unittest.main()
