import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import dataset_parsing.simulations_dataset as sds
from algorithms import ISBM
from validation.comparison import compare_metrics_dimensions, compare_result_dim, compare_time_samples, \
    compare_time_dimensions, compare_time_graph_vs_array_structure, compare_result_graph_vs_array_structure, \
    compare_metrics_graph_vs_array_structure, try_metric

import dataset_parsing.clustering_datasets as cds
import visualization.scatter_plot as sp

os.chdir('../')

X, y = sds.get_dataset_simulation_pca_2d(4)

pn=10
sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
sp.plot_grid_cm_tab(f'ISBM (PN={pn}) on Sim4', X, pn, sbm_graph_labels, marker='o')

pn=25
sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
sp.plot_grid_cm_tab(f'ISBM (PN={pn}) on Sim4', X, pn, sbm_graph_labels, marker='o')

plt.show()