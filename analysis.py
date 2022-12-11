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




# #### METRIC ANALYSIS ####
# no_noise=False
# X, y = cds.generate_simulated_data()
# try_metric(X, y, 6, 0.5, 25)
# compare_result_graph_vs_array_structure('UO', X, y, 6, 0.5, 25)
#
# X, y = sds.get_dataset_simulation_pca_2d(4)
# try_metric(X, y, 5, 0.1, 25)
# compare_result_graph_vs_array_structure('Sim4', X, y, 5, 0.1, 10)
#
# X, y = sds.get_dataset_simulation_pca_2d(1)
# try_metric(X, y, 17, 0.05, 46)
# compare_result_graph_vs_array_structure('Sim1', X, y, 17, 0.05, 46)
#
# X, y = sds.get_dataset_simulation_pca_2d(22)
# try_metric(X, y, 7, 0.05, 46)
# compare_result_graph_vs_array_structure('Sim22', X, y, 7, 0.1, 46)
#
# X, y = sds.get_dataset_simulation_pca_2d(21)
# try_metric(X, y, 5, 0.1, 20)
# compare_result_graph_vs_array_structure('Sim21', X, y, 5, 0.1, 20)
#
# X, y = sds.get_dataset_simulation_pca_2d(30)
# try_metric(X, y, 6, 0.1, 40)
# compare_result_graph_vs_array_structure('Sim30', X, y, 6, 0.1, 40)
#
#
### BY DATASET ANALYSIS #####

X, y = cds.generate_simulated_data()
compare_result_graph_vs_array_structure('UO', X, y, 6, 0.5)
compare_time_graph_vs_array_structure(X, y, 6, 0.5, 25, runs=100)
compare_metrics_graph_vs_array_structure('UO', X, y, 6, 0.5, 25)

X, y = sds.get_dataset_simulation_pca_2d(4)
print(len(np.unique(y)))
compare_result_graph_vs_array_structure('Sim4', X, y, 5, 0.1, 25)
compare_time_graph_vs_array_structure(X, y, 5, 0.1, 25, runs=100)
compare_metrics_graph_vs_array_structure('Sim4', X, y, 5, 0.1, 25)

X, y = sds.get_dataset_simulation_pca_2d(1)
print(len(np.unique(y)))
compare_result_graph_vs_array_structure('Sim1', X, y, 17, 0.1, 46)
compare_time_graph_vs_array_structure(X, y, 17, 0.1, 46, runs=100)
compare_metrics_graph_vs_array_structure('Sim1', X, y, 17, 0.05, 46)

X, y = sds.get_dataset_simulation_pca_2d(22)
print(len(np.unique(y)))
compare_result_graph_vs_array_structure('Sim22', X, y, 7, 0.05, 46)
compare_time_graph_vs_array_structure(X, y, 7, 0.1, 46, runs=100)
compare_metrics_graph_vs_array_structure('Sim22', X, y, 7, 0.05, 46)

X, y = sds.get_dataset_simulation_pca_2d(21)
print(len(np.unique(y)))
compare_result_graph_vs_array_structure('Sim21', X, y, 5, 0.1, 20)
compare_time_graph_vs_array_structure(X, y, 5, 0.1, 20, runs=100)
compare_metrics_graph_vs_array_structure('Sim21', X, y, 5, 0.1, 20)

X, y = sds.get_dataset_simulation_pca_2d(30)
compare_result_graph_vs_array_structure('Sim30', X, y, 6, 0.1, 40)
compare_time_graph_vs_array_structure(X, y, 6, 0.5, 40, runs=100)
compare_metrics_graph_vs_array_structure('Sim30', X, y, 6, 0.1, 40)



### OVERALL ANALYSIS ####
# X, y = sds.get_dataset_simulation(4)
# print(len(np.unique(y)))
# compare_time_dimensions(2, 25)
# compare_time_dimensions(3, 25)
# compare_time_dimensions(4, 25)
# compare_time_dimensions(5, 25)
# compare_time_dimensions(6, 25)
# compare_time_dimensions(7)
# compare_time_dimensions(8)
# compare_metrics_dimensions('Sim4 - 2d', 2, 5, 0.1, 10)
# compare_metrics_dimensions('Sim4 - 3d', 3, 5, 0.25, 12)
# compare_metrics_dimensions('Sim4 - 4d', 4, 5, 0.4, 8)
# pca_nd = PCA(n_components=2).fit_transform(X)
# try_metric(pca_nd, y, 5, 0.1, 10)
# pca_nd = PCA(n_components=3).fit_transform(X)
# try_metric(pca_nd, y, 5, 0.1, 8)
# pca_nd = PCA(n_components=4).fit_transform(X)
# try_metric(pca_nd, y, 5, 0.1, 6)

# compare_metrics_dimensions(5, 5, 0.4)
# compare_metrics_dimensions(6, 5, 0.4)
# compare_result_dim(X, y, 2, 5, 0.1)
# compare_result_dim(X, y, 3, 5, 0.25)
# compare_result_dim(X, y, 4, 5, 0.4)


# compare_time_samples()






# import visualization.scatter_plot as sp
# from visualization.scatter_plot_additionals import plot_spikes_by_clusters
#
# X, y = sds.get_dataset_simulation(1)
# pca_2d = PCA(n_components=2)
# pca_2d = pca_2d.fit_transform(X)

# sp.plot(f'Sim1 ground truth', pca_2d, y, marker='o', alpha=0.5)
# plot_spikes_by_clusters(X,  y)
#
# kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=0).fit(pca_2d)
# sp.plot(f'KMeans on Sim1', pca_2d, kmeans.labels_, marker='o', alpha=0.5)
# plot_spikes_by_clusters(X,  kmeans.labels_)
#
# isosplit_labels = np.loadtxt("./isosplit/sim1_isosplit.csv", delimiter=',', dtype=int)
# sp.plot(f'ISO-SPLIT on Sim1', pca_2d, isosplit_labels, marker='o', alpha=0.5)
# plot_spikes_by_clusters(X,  isosplit_labels)

# pn=46
# sbm_graph_labels = ISBM.run(pca_2d, pn, ccThreshold=5, adaptivePN=True)
# sp.plot(f'ISBM on Sim1', pca_2d, sbm_graph_labels, marker='o')
# plot_spikes_by_clusters(X,  sbm_graph_labels)
#
# plt.show()