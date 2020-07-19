import pytest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from src.utils import plot_roc_curve, plot_pr_curve


@pytest.fixture()
def sample_y_prob_y_true():
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    y_true = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    return y_prob, y_true


def test_plot_roc_curve(sample_y_prob_y_true):
    # verify that what is in the plot is what comes out of roc_curve
    y_prob = sample_y_prob_y_true[0]
    y_true = sample_y_prob_y_true[1]

    f, ax = plt.subplots()

    fpr, tpr, threshs = roc_curve(y_true, y_prob)

    plot_roc_curve(y_true, y_prob, show=False, verbose=False)

    x_plot, y_plot = ax.lines[0].get_xydata().T

    np.testing.assert_array_equal(x_plot, fpr)
    np.testing.assert_array_equal(y_plot, tpr)


def test_plot_pr_curve(sample_y_prob_y_true):
    # verify that what is in the plot is what comes out of
    # precision_recall_curve
    y_prob = sample_y_prob_y_true[0]
    y_true = sample_y_prob_y_true[1]

    f, ax = plt.subplots()

    precision_, recall_, threshs = precision_recall_curve(y_true, y_prob)

    plot_pr_curve(y_true, y_prob, show=False, verbose=False)

    x_plot, y_plot = ax.lines[0].get_xydata().T

    np.testing.assert_array_equal(x_plot, recall_)
    np.testing.assert_array_equal(y_plot, precision_)
