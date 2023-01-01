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
from visualization.scatter_plot_additionals import plot_spikes_by_clusters, plot_spikes_by_clusters_cmap

os.chdir('../')

X, y = sds.get_dataset_simulation(1)
pca_2d = PCA(n_components=2)
pca_2d = pca_2d.fit_transform(X)

# sp.plot_cm_tab(f'Sim1 ground truth', pca_2d, y, marker='o', alpha=0.7, cmap='tab20')
# plot_spikes_by_clusters_cmap(X,  y)

kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=0).fit(pca_2d)
sp.plot_cm_tab(f'KMeans on Sim1', pca_2d, kmeans.labels_, marker='o', alpha=0.7, cmap='tab20')
# plot_spikes_by_clusters_cmap(X,  kmeans.labels_)
#
# isosplit_labels = np.loadtxt("./isosplit/sim1_isosplit.csv", delimiter=',', dtype=int)
# sp.plot_cm_tab(f'ISO-SPLIT on Sim1', pca_2d, isosplit_labels, marker='o', alpha=0.7, cmap='tab20')
# plot_spikes_by_clusters_cmap(X,  isosplit_labels)
#
pn=46
sbm_graph_labels = ISBM.run(pca_2d, pn, ccThreshold=5, adaptivePN=True)
sp.plot_cm_tab(f'ISBM on Sim1', pca_2d, sbm_graph_labels, marker='o', alpha=0.7, cmap='tab20')
plot_spikes_by_clusters_cmap(X,  sbm_graph_labels)

plt.show()