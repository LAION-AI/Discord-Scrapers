import unittest
from helpers import starts_with_quotes, get_start_end_quotes


class TestHelpers(unittest.TestCase):

    def test_starts_with_quotes(self):
        self.assertTrue(starts_with_quotes('“Hello World”'))
        self.assertFalse(starts_with_quotes('Hello World'))
        self.assertFalse(starts_with_quotes(''))

    def test_get_start_end_quotes(self):
        #   Position helper:
        #   “Hello World”
        #   0123456789012
        self.assertEqual(get_start_end_quotes('“Hello World”'), (0, 12))
        self.assertEqual(get_start_end_quotes('Hello World'), (-1, -1))
        self.assertEqual(get_start_end_quotes('Hello “World”'), (6, 12))
        self.assertEqual(get_start_end_quotes('“Hello” World'), (0, 6))
        self.assertEqual(get_start_end_quotes('“Hello World'), (0, -1))
        self.assertEqual(get_start_end_quotes('“Hello World”"'), (0, 13))

    def test_substring_from_quotes(self):
        start, end = get_start_end_quotes('“Hello World”"')
        self.assertEqual('Hello World”', '“Hello World”"'[start+1:end])


if __name__ == '__main__':
    unittest.main()
