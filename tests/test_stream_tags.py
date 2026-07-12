import unittest

from evaluation.stream_tags import (
    count_stream_lines,
    extract_stream_lines,
    latest_stream_line_closed,
    stream_tag_sequence_reward,
    stream_target_reached,
    trim_to_stream_lines,
)


class StreamTagTests(unittest.TestCase):
    def test_accepts_decreasing_countdown_convention(self):
        text = "\n".join(f"[r:{i}/{3 - i}][V:1]C|" for i in range(4))
        lines = extract_stream_lines(text)

        self.assertEqual(count_stream_lines(text), 4)
        self.assertEqual(stream_tag_sequence_reward(lines), 1.0)

    def test_accepts_one_based_index_total_convention(self):
        text = "\n".join(f"[r:{i}/4][V:1]C|" for i in range(1, 5))
        lines = extract_stream_lines(text)

        self.assertEqual(count_stream_lines(text), 4)
        self.assertEqual(stream_tag_sequence_reward(lines), 1.0)

    def test_accepts_zero_based_index_last_convention(self):
        text = "\n".join(f"[r:{i}/3][V:1]C|" for i in range(4))
        lines = extract_stream_lines(text)

        self.assertEqual(count_stream_lines(text), 4)
        self.assertEqual(stream_tag_sequence_reward(lines), 1.0)

    def test_penalizes_skipped_stream_index(self):
        text = "[r:1/4][V:1]C|\n[r:3/4][V:1]D|\n"

        self.assertLess(stream_tag_sequence_reward(extract_stream_lines(text)), 1.0)

    def test_target_reached_requires_closed_latest_stream_line(self):
        open_text = "[r:1/2][V:1]C|\n[r:2/2][V:1]D"
        closed_text = open_text + "|"

        self.assertFalse(stream_target_reached(open_text, 2))
        self.assertTrue(stream_target_reached(closed_text, 2))
        self.assertTrue(latest_stream_line_closed(closed_text))

    def test_trim_keeps_metadata_and_requested_stream_lines(self):
        text = "M:3/4\n[r:1/3][V:1]C|\n[r:2/3][V:1]D|\n[r:3/3][V:1]E|\n"

        self.assertEqual(trim_to_stream_lines(text, 2), "M:3/4\n[r:1/3][V:1]C|\n[r:2/3][V:1]D|\n")


if __name__ == "__main__":
    unittest.main()
