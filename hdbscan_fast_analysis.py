import math
import os

from hdbscan import HDBSCAN
from joblib import Memory
from matplotlib import pyplot as plt

import visualization.scatter_plot as sp

from validation.comparison import get_metrics

from validation.metric import ss_metric


import dataset_parsing.simulations_dataset as sds
import dataset_parsing.clustering_datasets as cds

import numpy as np
from sklearn.decomposition import PCA

from dataset_parsing.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, plot_multitrode
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel

import time


# X, y = cds.generate_simulated_data()
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#         gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
#         metric='euclidean', min_cluster_size=10, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on UO', X, hdbscan_labels, marker='o')
#
#
#
#
#
#
# X, y = sds.get_dataset_simulation_pca_2d(4)
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
#                       metric='euclidean', min_cluster_size=10, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on Sim4', X, hdbscan_labels, marker='o')
#
#
#
#
#
#
#
#
# X, y = sds.get_dataset_simulation_pca_2d(1)
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=10, memory=Memory(cachedir=None),
#                       metric='euclidean', min_cluster_size=15, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on Sim1', X, hdbscan_labels, marker='o')
#
#
#
#
#
#
#
# X, y = sds.get_dataset_simulation_pca_2d(22)
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=5, memory=Memory(cachedir=None),
#                       metric='euclidean', min_cluster_size=30, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# hdbscan_labels = hdbscan.labels_
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on Sim22', X, hdbscan_labels, marker='o')
#
#
#
#
#
#
#
#
# X, y = sds.get_dataset_simulation_pca_2d(21)
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
#                       metric='euclidean', min_cluster_size=5, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on Sim21', X, hdbscan_labels, marker='o')
#
#
#
#
#
#
# X, y = sds.get_dataset_simulation_pca_2d(30)
# time_sum = 0
# for i in range(0, 100):
#     start = time.time()
#     hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
#                       metric='euclidean', min_cluster_size=50, min_samples=None, p=None).fit(X)
#     hdbscan_labels = hdbscan.labels_
#     time_sum +=(time.time() - start)
# print(time_sum/100)
#
# # print(ss_metric(y, hdbscan_labels))
# # get_metrics(y, hdbscan_labels)
# sp.plot(f'HDBSCAN on Sim30', X, hdbscan_labels, marker='o')
#
# plt.show()





# for nr_comp in [3,4,5,6]:
#     data, y = sds.get_dataset_simulation(4)
#     pca_ = PCA(n_components=nr_comp)
#     spikes_pca_ = pca_.fit_transform(data)
#
#     time_sum = 0
#     for i in range(0, 100):
#         start = time.time()
#         hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                           gen_min_span_tree=False, leaf_size=10, memory=Memory(cachedir=None),
#                           metric='euclidean', min_cluster_size=30, min_samples=None, p=None).fit(spikes_pca_)
#         hdbscan_labels = hdbscan.labels_
#         time_sum +=(time.time() - start)
#     print(time_sum/100)
#
#     print(ss_metric(y, hdbscan_labels))
#     # get_metrics(y, hdbscan_labels)
#     print()

# for j in range(1, 11):
#     X, y = cds.generate_simulated_data(j * 250)
#     time_sum = 0
#     for i in range(0, 100):
#         start = time.time()
#         hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#             gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
#             metric='euclidean', min_cluster_size=10, min_samples=None, p=None).fit(X)
#         hdbscan_labels = hdbscan.labels_
#         time_sum +=(time.time() - start)
#     print(time_sum/100)
#
#     # print(ss_metric(y, hdbscan_labels))
#     # get_metrics(y, hdbscan_labels)
#     sp.plot(f'HDBSCAN on UO', X, hdbscan_labels, marker='o')





DATASET_PATH = '../DATA/TINS/M017_0004_Tetrode_try2/ssd/'

spikes_per_unit, unit_multitrode, _ = parse_ssd_file(DATASET_PATH)
MULTITRODE_WAVEFORM_LENGTH = 232
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_MULTITRODES = 8
NR_ELECTRODES_PER_MULTITRODE = 4
MULTITRODE_CHANNEL = 7

timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)

waveforms = read_waveforms(waveform_file)

waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, MULTITRODE_WAVEFORM_LENGTH)

units_in_multitrode, labels = units_by_channel(unit_multitrode, waveforms_by_unit,
                                               data_length=MULTITRODE_WAVEFORM_LENGTH,
                                               number_of_channels=NR_MULTITRODES)
units_by_multitrodes = split_multitrode(units_in_multitrode, MULTITRODE_WAVEFORM_LENGTH, WAVEFORM_LENGTH)

labels = labels[MULTITRODE_CHANNEL]

data_electrode1 = units_by_multitrodes[MULTITRODE_CHANNEL][0]

pca_ = PCA(n_components=3)
pca_electrode1 = pca_.fit_transform(data_electrode1)
hdbscan = HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=10, memory=Memory(cachedir=None),
    metric='euclidean', min_cluster_size=10, min_samples=None, p=None).fit(pca_electrode1)
hdbscan_labels = hdbscan.labels_

sp.plot(f'HDBSCAN on Electrode 1', pca_electrode1, hdbscan_labels, marker='o', alpha=0.5)
plt.show()

print(ss_metric(labels, hdbscan_labels))
get_metrics(labels, hdbscan_labels)