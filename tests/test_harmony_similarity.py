import unittest

from evaluation.harmony_similarity import generic_dtw_alignment, generic_dtw_similarity


class HarmonySimilarityTests(unittest.TestCase):
    def test_generic_dtw_alignment_keeps_similarity_and_path(self):
        reference = [0, 7, 2]
        candidate = [0, 2]

        alignment = generic_dtw_alignment(
            reference,
            candidate,
            lambda left, right: 1.0 if left == right else 0.0,
            band_ratio=1.0,
        )

        self.assertEqual(alignment.similarity, generic_dtw_similarity(reference, candidate, lambda l, r: 1.0 if l == r else 0.0, band_ratio=1.0))
        self.assertEqual(len(alignment.path), len(alignment.local_similarities))
        self.assertGreater(len(alignment.path), 0)
        self.assertEqual(alignment.path[0], (0, 0))
        self.assertEqual(alignment.path[-1], (2, 1))


if __name__ == "__main__":
    unittest.main()
