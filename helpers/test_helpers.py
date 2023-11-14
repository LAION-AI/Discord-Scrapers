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
        self.assertEqual(get_start_end_quotes('"In "The Final Bow," we enter the twilight of a once-celebrated theater actor. The stage is a kaleidoscope of his illustrious career, fragments of his most iconic roles haunting the wings. Each wrinkle on his time-worn face tells a story, eyes shimmering with the ghosts of a thousand characters. Tonight, the spotlight embraces him one last time, casting long shadows that blend with the encroaching darkness of the wings. As the curtain falls, His legacy is etched not in the applause, but in the silence that follows—the hushed reverence for a lifetime devoted to the craft. The theater, once a vessel of roaring life, now whispers with the echoes of his final bow, a poignant testament to the ephemeral beauty of performance and the enduring power of storytelling."'), (0, 768))
        self.assertEqual(get_start_end_quotes('""Test" starting and ending "quotes""'), (0, 36))
                         
    def test_substring_from_quotes(self):
        start, end = get_start_end_quotes('“Hello World”"')
        self.assertEqual('Hello World”', '“Hello World”"'[start+1:end])


if __name__ == '__main__':
    unittest.main()
