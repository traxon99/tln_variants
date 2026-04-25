"""Smoke tests for train/train_utils.py."""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_utils import normalize_speed, split_dataset, huber_loss_np


class TestNormalizeSpeed:
    def test_range(self):
        speed = np.array([0.0, 2.5, 5.0])
        out, max_s = normalize_speed(speed)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_explicit_max(self):
        speed = np.array([0.0, 1.0, 2.0])
        out, max_s = normalize_speed(speed, max_speed=4.0)
        np.testing.assert_allclose(out, [0.0, 0.25, 0.5])
        assert max_s == 4.0

    def test_zero_range_raises(self):
        with pytest.raises(ValueError):
            normalize_speed(np.array([3.0, 3.0, 3.0]))


class TestSplitDataset:
    def setup_method(self):
        np.random.seed(0)
        n = 100
        self.lidar = np.random.rand(n, 20)
        self.servo = np.random.rand(n)
        self.speed = np.random.rand(n)

    def test_split_sizes(self):
        X_tr, X_te, y_tr, y_te = split_dataset(self.lidar, self.servo, self.speed, 0.8)
        assert len(X_tr) == 80
        assert len(X_te) == 20

    def test_label_shape(self):
        _, _, y_tr, _ = split_dataset(self.lidar, self.servo, self.speed, 0.85)
        assert y_tr.shape[1] == 2  # [servo, speed]

    def test_no_overlap(self):
        X_tr, X_te, _, _ = split_dataset(self.lidar, self.servo, self.speed, 0.8)
        # Row hashes must be disjoint
        tr_set = set(map(tuple, X_tr.tolist()))
        te_set = set(map(tuple, X_te.tolist()))
        assert len(tr_set & te_set) == 0


class TestHuberLossNp:
    def test_perfect_prediction(self):
        y = np.array([[1.0, 0.5], [0.2, 0.8]])
        assert huber_loss_np(y, y) == pytest.approx(0.0)

    def test_small_error_is_quadratic(self):
        # For |error| <= 1, loss = 0.5 * error^2
        y_true = np.array([[0.0]])
        y_pred = np.array([[0.5]])
        expected = 0.5 * 0.5 ** 2
        assert huber_loss_np(y_true, y_pred) == pytest.approx(expected)

    def test_large_error_is_linear(self):
        # For |error| > 1, loss = delta * (|error| - 0.5 * delta)
        y_true = np.array([[0.0]])
        y_pred = np.array([[3.0]])  # error = 3, delta = 1
        expected = 1.0 * (3.0 - 0.5)
        assert huber_loss_np(y_true, y_pred) == pytest.approx(expected)


class TestLoadCsvDatasetErrors:
    def test_missing_file_raises(self):
        from train_utils import load_csv_dataset
        with pytest.raises(FileNotFoundError):
            load_csv_dataset(['/nonexistent/path/data.csv'])
