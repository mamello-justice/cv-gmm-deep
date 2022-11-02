import unittest
import numpy as np

from cv_gmm_deep.common import split_data


class TestSplitData(unittest.TestCase):
    def test_split_data(self):
        x = y = np.arange(20)

        ratios = [0.7, 0.15, 0.15]

        out = split_data(x, y, ratios)

        self.assertEqual(len(out), len(ratios))
        self.assertEqual(len(out[0][0]), 14)
        self.assertEqual(len(out[1][0]), 3)
        self.assertEqual(len(out[2][0]), 3)
