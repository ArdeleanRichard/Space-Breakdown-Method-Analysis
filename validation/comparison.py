import math

import numpy as np
import time

from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans, estimate_bandwidth, MeanShift, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, v_measure_score
from fcmeans import FCM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from algorithms import ISBM, SBM
import dataset_parsing.simulations_dataset as sds
from visualization import scatter_plot as sp
from validation.scores import purity_score
from validation.metric import ss_metric

import dataset_parsing.clustering_datasets as cds




def try_metric(X, y, n_clusters, eps, pn=25, no_noise=True):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5)

    # sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5)

    sbm_graph2_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)

    bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)

    print(f"KMeans: {ss_metric(y, kmeans.labels_):.3f}")
    print(f"DBSCAN: {ss_metric(y, dbscan.labels_):.3f}")
    print(f"MS: {ss_metric(y, ms.labels_):.3f}")
    print(f"AC: {ss_metric(y, ward.labels_):.3f}")
    print(f"FCM: {ss_metric(y, labels):.3f}")
    print(f"SBMog: {ss_metric(y, sbm_array_labels):.3f}")
    # print(f"ISBM: {ss_metric(y, sbm_graph_labels):.3f}")
    print(f"ISBM: {ss_metric(y, sbm_graph2_labels):.3f}")
    # test_labels = np.array(list(range(0, len(y))))
    # print(f"Test: {ss_metric(y, test_labels):.3f}")
    print()
    #
    # print()
    # print(f"KMeans: {ss_metric_unweighted(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_unweighted(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_unweighted(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_unweighted(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_unweighted(y, test_labels):.3f}")
    #
    # print()
    # print(f"KMeans: {ss_metric_unweighted2(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_unweighted2(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_unweighted2(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_unweighted2(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_unweighted2(y, test_labels):.3f}")
    # print()
    #
    # print()
    # print(f"KMeans: {ss_metric_trial1(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_trial1(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_trial1(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_trial1(y, sbm_graph2_labels, 0):.3f}")
    # # print(f"Test: {ss_metric_trial1(y, test_labels):.3f}")
    # print()
    #
    # print()
    # print(f"KMeans: {ss_metric_trial2(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_trial2(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_trial2(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_trial2(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_trial2(y, test_labels):.3f}")
    # print()




def compare_plots_and_metrics(Title, X, y, n_clusters, eps, pn=25, pn2=None, Xvis=None):
    sp.plot(f'{Title} ground truth', X, y, marker='o', alpha=0.3)

    if Xvis is None:
        Xvis = np.copy(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    sp.plot(f'KMeans on {Title}', Xvis, kmeans.labels_, marker='o', alpha=0.5)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
    sp.plot(f'DBSCAN on {Title}', Xvis, dbscan.labels_, marker='o', alpha=0.5)
    #
    # # af = AffinityPropagation(random_state=5).fit(X)
    # # sp.plot(f'AffinityPropagation on {Title}', X, af.labels_, marker='o')
    #
    bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    sp.plot(f'MeanShift on {Title}', Xvis, ms.labels_, marker='o', alpha=0.5)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)
    sp.plot(f'AgglomerativeClustering on {Title}', X, ward.labels_, marker='o', alpha=0.5)

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)
    sp.plot(f'FCM on {Title}', Xvis, labels, marker='o', alpha=0.5)

    if pn2 is None:
        sbm_array_labels = SBM.best(X, pn, ccThreshold=5)
    else:
        sbm_array_labels = SBM.best(X, pn2, ccThreshold=5)

    # sp.plot_grid(f'SBM on {Title}', X, pn, sbm_array_labels, marker='o')
    sp.plot(f'SBM on {Title}', Xvis, sbm_array_labels, marker='o', alpha=0.5)

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
    # sp.plot_grid(f'ISBM on {Title}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)
    sp.plot(f'ISBM on {Title}', Xvis, sbm_graph_labels, marker='o', alpha=0.5)

    plt.show()


    print(f"{Title} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_rand_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_rand_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_rand_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_rand_score(y, labels):.3f}\t"
          f"SBM={adjusted_rand_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={adjusted_rand_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_mutual_info_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_mutual_info_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_mutual_info_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_mutual_info_score(y, labels):.3f}\t"
          f"SBM={adjusted_mutual_info_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={adjusted_mutual_info_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={purity_score(y, dbscan.labels_):.3f}\t"
          f"MS={purity_score(y, ms.labels_):.3f}\t"
          f"Ag={purity_score(y, ward.labels_):.3f}\t"
          f"FCM={purity_score(y, labels):.3f}\t"
          f"SBM={purity_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={purity_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={fowlkes_mallows_score(y, dbscan.labels_):.3f}\t"
          f"MS={fowlkes_mallows_score(y, ms.labels_):.3f}\t"
          f"Ag={fowlkes_mallows_score(y, ward.labels_):.3f}\t"
          f"FCM={fowlkes_mallows_score(y, labels):.3f}\t"
          f"SBM={fowlkes_mallows_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={fowlkes_mallows_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={v_measure_score(y, dbscan.labels_):.3f}\t"
          f"MS={v_measure_score(y, ms.labels_):.3f}\t"
          f"Ag={v_measure_score(y, ward.labels_):.3f}\t"
          f"FCM={v_measure_score(y, labels):.3f}\t"
          f"SBM={v_measure_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={v_measure_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - SCS: "
          f"KMeans={ss_metric(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={ss_metric(y, dbscan.labels_):.3f}\t"
          f"MS={ss_metric(y, ms.labels_):.3f}\t"
          f"Ag={ss_metric(y, ward.labels_):.3f}\t"
          f"FCM={ss_metric(y, labels):.3f}\t"
          f"SBM={ss_metric(y, sbm_array_labels):.3f}\t"
          f"ISBM={ss_metric(y, sbm_graph_labels):.3f}\t")



def compare_result_graph_vs_array_structure(Title, X, y, n_clusters, eps, pn=25, pn2=None, Xvis=None):
    sp.plot(f'Synthetic dataset ({Title})  ground truth', X, y, marker='o', alpha=0.3)

    if Xvis is None:
        Xvis = np.copy(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    sp.plot(f'KMeans on {Title}', Xvis, kmeans.labels_, marker='o')

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
    sp.plot(f'DBSCAN on {Title}', Xvis, dbscan.labels_, marker='o')
    #
    # # af = AffinityPropagation(random_state=5).fit(X)
    # # sp.plot(f'AffinityPropagation on {Title}', X, af.labels_, marker='o')
    #
    bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    sp.plot(f'MeanShift on {Title}', Xvis, ms.labels_, marker='o')

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)
    sp.plot(f'AgglomerativeClustering on {Title}', X, ward.labels_, marker='o')

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)
    sp.plot(f'FCM on {Title}', Xvis, labels, marker='o')

    if pn2 is None:
        sbm_array_labels = SBM.best(X, pn, ccThreshold=5)
    else:
        sbm_array_labels = SBM.best(X, pn2, ccThreshold=5)

    # sp.plot_grid(f'SBM on {Title}', X, pn, sbm_array_labels, marker='o')
    sp.plot(f'SBM on {Title}', Xvis, sbm_array_labels, marker='o')

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
    # sp.plot_grid(f'ISBM on {Title}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)
    sp.plot(f'ISBM on {Title}', Xvis, sbm_graph_labels, marker='o')

    plt.show()


def compare_time_graph_vs_array_structure(X, y, n_clusters, eps, pn = 10, runs=25):
    print(len(X))


    kmeans_time = 0
    dbscan_time = 0
    sbm_array_time = 0
    sbm_graph_time = 0
    sbm_graph2_time = 0
    ms_time = 0
    ac_time = 0
    fcm_time = 0

    for i in range(runs):
        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        kmeans_time += (time.time() - start)

        # start = time.time()
        # dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
        # dbscan_time += (time.time() - start)

        start = time.time()
        sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5)
        sbm_array_time += (time.time() - start)

        # start = time.time()
        # sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5)
        # sbm_graph_time += (time.time() - start)

        start = time.time()
        bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
        ms_time += (time.time() - start)

        start = time.time()
        ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)
        ac_time += (time.time() - start)

        start = time.time()
        my_model = FCM(n_clusters=n_clusters)
        my_model.fit(X)
        labels = my_model.predict(X)
        fcm_time += (time.time() - start)

        start = time.time()
        sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
        sbm_graph2_time += (time.time() - start)

    print(f"Kmeans time - {runs} runs: {kmeans_time/runs:.3f}s")
    print(f"DBSCAN time - {runs} runs: {dbscan_time/runs:.3f}s")
    print(f"SBM array time - {runs} runs: {sbm_array_time/runs:.3f}s")
    print(f"SBM graph time - {runs} runs: {sbm_graph_time/runs:.3f}s")
    print(f"SBM graph2 time - {runs} runs: {sbm_graph2_time/runs:.3f}s")
    print(f"MS time - {runs} runs: {ms_time/runs:.3f}s")
    print(f"AC time - {runs} runs: {ac_time/runs:.3f}s")
    print(f"FCM time - {runs} runs: {fcm_time/runs:.3f}s")


def get_metrics(y, labels, method='ISO-SPLIT'):
    print(f" ARI: "
          f"{method}={adjusted_rand_score(y, labels):.3f}\t")

    print(f"AMI: "
          f"{method}={adjusted_mutual_info_score(y, labels):.3f}\t")

    print(f"Purity: "
          f"{method}={purity_score(y, labels):.3f}\t" )

    print(f"FMI: "
          f"{method}={fowlkes_mallows_score(y, labels):.3f}\t")

    print(f"VM: "
          f"{method}={v_measure_score(y, labels):.3f}\t")
    print()



def compare_metrics_graph_vs_array_structure(Data, X, y, n_clusters, eps, pn=25):
    # dataset

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5)

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5)
    sbm_graph2_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)

    bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)

    #metric - ARI
    print(f"{Data} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_rand_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_rand_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_rand_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_rand_score(y, labels):.3f}\t"
          # f"SBM_graph={adjusted_rand_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_rand_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_mutual_info_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_mutual_info_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_mutual_info_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_mutual_info_score(y, labels):.3f}\t"
          f"SBM_array={adjusted_mutual_info_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={adjusted_mutual_info_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_mutual_info_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={purity_score(y, dbscan.labels_):.3f}\t"
          f"MS={purity_score(y, ms.labels_):.3f}\t"
          f"Ag={purity_score(y, ward.labels_):.3f}\t"
          f"FCM={purity_score(y, labels):.3f}\t"
          f"SBM_array={purity_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={purity_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={purity_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={fowlkes_mallows_score(y, dbscan.labels_):.3f}\t"
          f"MS={fowlkes_mallows_score(y, ms.labels_):.3f}\t"
          f"Ag={fowlkes_mallows_score(y, ward.labels_):.3f}\t"
          f"FCM={fowlkes_mallows_score(y, labels):.3f}\t"
          f"SBM_array={fowlkes_mallows_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={fowlkes_mallows_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={fowlkes_mallows_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={v_measure_score(y, dbscan.labels_):.3f}\t"
          f"MS={v_measure_score(y, ms.labels_):.3f}\t"
          f"Ag={v_measure_score(y, ward.labels_):.3f}\t"
          f"FCM={v_measure_score(y, labels):.3f}\t"
          f"SBM_array={v_measure_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={v_measure_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={v_measure_score(y, sbm_graph2_labels):.3f}\t")

    # print(f"UO - SS: "
    #       f"KMeans={silhouette_score(X, kmeans.labels_):.3f}\t"
    #       f"DBSCAN={silhouette_score(X, dbscan.labels_):.3f}\t"
    #       f"SBM_array={silhouette_score(X, sbm_array_labels):.3f}\t"
    #       # f"SBM_graph={silhouette_score(X, sbm_graph_labels):.3f}\t"
    #       f"SBM_graph2={silhouette_score(X, sbm_graph2_labels):.3f}\t")
    #
    # print(f"UO - CHS: "
    #       f"KMeans={calinski_harabasz_score(X, kmeans.labels_):.3f}\t"
    #       f"DBSCAN={calinski_harabasz_score(X, dbscan.labels_):.3f}\t"
    #       f"SBM_array={calinski_harabasz_score(X, sbm_array_labels):.3f}\t"
    #       # f"SBM_graph={calinski_harabasz_score(X, sbm_graph_labels):.3f}\t"
    #       f"SBM_graph2={calinski_harabasz_score(X, sbm_graph2_labels):.3f}\t")
    #
    # print(f"UO - DBS: "
    #       f"KMeans={davies_bouldin_score(X, kmeans.labels_):.3f}\t"
    #       f"DBSCAN={davies_bouldin_score(X, dbscan.labels_):.3f}\t"
    #       f"SBM_array={davies_bouldin_score(X, sbm_array_labels):.3f}\t"
    #       # f"SBM_graph={davies_bouldin_score(X, sbm_graph_labels):.3f}\t"
    #       f"SBM_graph2={davies_bouldin_score(X, sbm_graph2_labels):.3f}\t")

    print()


def compare_time_dimensions(nr_dim=3, runs=25):
    X, y = sds.get_dataset_simulation(4)
    pca_3d = PCA(n_components=nr_dim)
    X = pca_3d.fit_transform(X)

    compare_time_graph_vs_array_structure(X, y, 5, 0.1, runs)


def compare_time_samples():
    for i in range(6, 15):
        size = i * 250
        X, y = cds.generate_simulated_data(size)
        compare_time_graph_vs_array_structure(X, y, 6, 0.5, 100)


def compare_metrics_dimensions(Data, nr_dim, n_clusters, eps, pn=25):
    X, y = sds.get_dataset_simulation(4)
    pca_nd = PCA(n_components=nr_dim)
    X = pca_nd.fit_transform(X)

    print(nr_dim)
    compare_metrics_graph_vs_array_structure(Data, X, y, n_clusters, eps, pn)


def compare_result_dim(X, y, nr_dim, n_clusters, eps):
    pca_2d = PCA(n_components=2)
    X2D = pca_2d.fit_transform(X)

    pca_nd = PCA(n_components=nr_dim)
    X = pca_nd.fit_transform(X)

    pn = 25
    sp.plot('Synthetic dataset (UO) ground truth', X2D, y, marker='o', alpha=0.5)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    sp.plot('KMeans on UO', X2D, kmeans.labels_, marker='o')

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
    sp.plot('DBSCAN on UO', X2D, dbscan.labels_, marker='o')

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5)
    sp.plot_grid('SBM array on UO', X2D, pn, sbm_array_labels, marker='o')

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5)
    sp.plot_grid('SBM graph on UO', X2D, pn, sbm_graph_labels, marker='o')

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
    sp.plot_grid('SBM graph2 on UO', X2D, pn, sbm_graph_labels, marker='o', adaptivePN=True)

    plt.show()