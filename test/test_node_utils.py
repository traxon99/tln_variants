"""Smoke tests for tln_variants/node_utils.py."""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tln_variants'))
from node_utils import linear_map, preprocess_scan, CLIP_MAX, NOISE_STD


class TestLinearMap:
    def test_identity(self):
        assert linear_map(0.5, 0.0, 1.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_rescale(self):
        assert linear_map(5.0, 0.0, 10.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_invert(self):
        assert linear_map(0.0, -1.0, 1.0, 1.0, -1.0) == pytest.approx(0.0)

    def test_zero_range_guard(self):
        result = linear_map(3.0, 3.0, 3.0, 0.0, 10.0)
        assert result == pytest.approx(5.0)

    def test_numpy_array(self):
        arr = np.array([0.0, 5.0, 10.0])
        out = linear_map(arr, 0.0, 10.0, 0.0, 1.0)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


class TestPreprocessScan:
    def test_output_length(self):
        ranges = [1.0] * 1080
        out = preprocess_scan(ranges, 540)
        assert len(out) == 540

    def test_nan_replaced(self):
        ranges = [float('nan'), float('inf'), float('-inf'), 1.0]
        out = preprocess_scan(ranges, 4)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_clip_applied(self):
        ranges = [0.5, CLIP_MAX + 5.0, 2.0]
        out = preprocess_scan(ranges, 3)
        assert float(out[1]) <= CLIP_MAX

    def test_output_dtype_float32(self):
        ranges = [1.0, 2.0, 3.0]
        out = preprocess_scan(ranges, 3)
        assert out.dtype == np.float32

    def test_no_noise_by_default(self):
        ranges = [1.0] * 100
        out1 = preprocess_scan(ranges, 100, add_noise=False)
        out2 = preprocess_scan(ranges, 100, add_noise=False)
        np.testing.assert_array_equal(out1, out2)
