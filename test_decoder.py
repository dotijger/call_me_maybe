import unittest
import numpy as np

from new.setup import ConstrainedDecoder
from new.classes import Vocab

TINY_VOCAB = {0: '"', 1: "a", 2: "b", 3: "{", 4: "}"}


class TestConstrainedDecoderVocabMethods(unittest.TestCase):
    def setUp(self):
        self.decoder = ConstrainedDecoder.model_construct(
            llm_vocab=Vocab(
                vocab=TINY_VOCAB, inverted={v: k for k, v in TINY_VOCAB.items()}
            )
        )

    def test_find_match_hits(self):
        cases = [('"', 0), ("a", 1), ("{", 3), ("}", 4)]
        for token, expected_id in cases:
            with self.subTest(token=token):
                self.assertEqual(self.decoder._find_match(token), expected_id)

    def test_find_match_miss_returns_negative_one(self):
        self.assertEqual(self.decoder._find_match("zzz"), -1)

    def test_is_prefix(self):
        cases = [
            ("a", "abc", True),
            ("ab", "abc", True),
            ("abc", "abc", True),
            ("x", "abc", False),
        ]
        for small, big, expected in cases:
            with self.subTest(small=small, big=big):
                self.assertEqual(self.decoder._is_prefix(small, big), expected)

    def test_get_mask_shape_and_values(self):
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        mask = self.decoder._get_mask(logits, [1, 3])
        self.assertEqual(mask[1], 0)
        self.assertEqual(mask[3], 0)
        self.assertEqual(mask[0], -np.inf)
        self.assertEqual(mask[2], -np.inf)
        self.assertEqual(mask[4], -np.inf)


if __name__ == "__main__":
    unittest.main()
