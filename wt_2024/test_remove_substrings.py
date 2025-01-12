import unittest
from generate_best_ngrams import remove_substrings

# python -m unittest test_remove_substrings.py

class TestRemoveSubstrings(unittest.TestCase):
    def test_remove_substrings_basic(self):
        input_set = {"hello", "hell", "he", "hello world"}
        expected_output = {"he"}
        self.assertSetEqual(set(remove_substrings(input_set)), expected_output)

    def test_remove_substrings_empty(self):
        input_set = set()
        expected_output = set()
        self.assertSetEqual(set(remove_substrings(input_set)), expected_output)

    def test_remove_substrings_no_substrings(self):
        input_set = {"apple", "banana", "cherry"}
        expected_output = {"apple", "banana", "cherry"}
        self.assertSetEqual(set(remove_substrings(input_set)), expected_output)

    def test_remove_substrings_all_substrings(self):
        input_set = {"a", "ab", "abc", "abcd"}
        expected_output = {"a"}
        self.assertSetEqual(set(remove_substrings(input_set)), expected_output)

    def test_remove_substrings_mixed(self):
        input_set = {"a", "apple", "app", "appl", "banana", "ban"}
        expected_output = {"a"}
        self.assertSetEqual(set(remove_substrings(input_set)), expected_output)

if __name__ == '__main__':
    unittest.main()