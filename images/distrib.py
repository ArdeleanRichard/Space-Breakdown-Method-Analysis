import numpy as np
from matplotlib import pyplot as plt

from algorithms import ISBM
from algorithms.ISBM import data_preprocessing
from visualization import scatter_plot as sp
from visualization.label_map import LABEL_COLOR_MAP

import os
os.chdir("../")

avgPoints = 250
mu, sigma = 50, 10
mu2, sigma2 = 10, 40
mu3, sigma3 = -25, 10
X1 = np.random.normal(mu, sigma, (avgPoints, 2))
X2 = np.random.normal(mu2, sigma2, (avgPoints*2, 2))
X3 = np.random.normal(mu3, sigma3, (avgPoints, 2))
# X2 = np.random.uniform(mu2, sigma2, (avgPoints*2, 2))
X = np.concatenate([X1, X2, X3])
# X = np.concatenate([X1, X2])

c1Labels = np.full(len(X1), 1)
c2Labels = np.full(len(X2), 2)
c3Labels = np.full(len(X3), 3)

y = np.hstack((c1Labels, c2Labels, c3Labels))
# y = np.hstack((c1Labels, c2Labels))


def plot_all(X, y):
    plt.title("X axis histogram")
    plt.hist(X[:, 0], bins='auto')
    plt.show()

    plt.title("Y axis histogram")
    plt.hist(X[:, 1], bins='auto')
    plt.show()

    sp.plot("test", X, y, alpha=0.5)
    plt.show()

    pn=5
    labels = ISBM.run(X, pn=pn, ccThreshold=5, adaptivePN=True)
    sp.plot_grid('ISBM(PN=25) on Test Data with a Unimodal Distribution', X, pn, labels, marker='o', adaptivePN=True)
    plt.show()

    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(X, labels, pn, ax, ax_histx, ax_histy)
    plt.show()


def scatter_hist(X, labels, pn, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)


    # the scatter plot:
    X, pn = data_preprocessing(X, pn, adaptivePN=True)

    x = X[:, 0]
    y = X[:, 1]

    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    ax.scatter(x, y, marker='o', c=label_color, s=25, edgecolor='k')
    ax.grid(True)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    plt.gca()
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

plot_all(X, y)