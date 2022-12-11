import math
import os

from matplotlib import pyplot as plt

import visualization.scatter_plot as sp

from validation.comparison import get_metrics

from validation.metric import ss_metric


import dataset_parsing.simulations_dataset as sds

import numpy as np
from sklearn.decomposition import PCA

from dataset_parsing.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, plot_multitrode
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel


os.chdir("../")


# X, y = cds.generate_simulated_data()
# isosplit_labels = np.loadtxt("./isosplit/uo_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on UO', X, isosplit_labels, marker='o')
#
# X, y = sds.get_dataset_simulation_pca_2d(4)
# isosplit_labels = np.loadtxt("./isosplit/sim4_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on Sim4', X, isosplit_labels, marker='o')
#
# X, y = sds.get_dataset_simulation_pca_2d(1)
# isosplit_labels = np.loadtxt("./isosplit/sim1_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on Sim1', X, isosplit_labels, marker='o')
#
# X, y = sds.get_dataset_simulation_pca_2d(22)
# isosplit_labels = np.loadtxt("./isosplit/sim22_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on Sim22', X, isosplit_labels, marker='o')
#
# X, y = sds.get_dataset_simulation_pca_2d(21)
# isosplit_labels = np.loadtxt("./isosplit/sim21_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on Sim21', X, isosplit_labels, marker='o')
#
# X, y = sds.get_dataset_simulation_pca_2d(30)
# isosplit_labels = np.loadtxt("./isosplit/sim30_isosplit.csv", delimiter=',', dtype=int)
# print(ss_metric(y, isosplit_labels))
# get_metrics(y, isosplit_labels)
# sp.plot(f'ISO-SPLIT on Sim30', X, isosplit_labels, marker='o')
# plt.show()





for nr_comp in [3,4]:
    data, y = sds.get_dataset_simulation(4)
    isosplit_labels = np.loadtxt(f"./isosplit/sim4_{nr_comp}d_isosplit.csv", delimiter=',', dtype=int)
    print(ss_metric(y, isosplit_labels))
    get_metrics(y, isosplit_labels)








# DATASET_PATH = '../DATA/TINS/M017_0004_Tetrode_try2/ssd/'
#
# spikes_per_unit, unit_multitrode, _ = parse_ssd_file(DATASET_PATH)
# MULTITRODE_WAVEFORM_LENGTH = 232
# WAVEFORM_LENGTH = 58
# TIMESTAMP_LENGTH = 1
# NR_MULTITRODES = 8
# NR_ELECTRODES_PER_MULTITRODE = 4
# MULTITRODE_CHANNEL = 7
#
# timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
#
# waveforms = read_waveforms(waveform_file)
#
# waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, MULTITRODE_WAVEFORM_LENGTH)
#
# units_in_multitrode, labels = units_by_channel(unit_multitrode, waveforms_by_unit,
#                                                data_length=MULTITRODE_WAVEFORM_LENGTH,
#                                                number_of_channels=NR_MULTITRODES)
# units_by_multitrodes = split_multitrode(units_in_multitrode, MULTITRODE_WAVEFORM_LENGTH, WAVEFORM_LENGTH)
#
# labels = labels[MULTITRODE_CHANNEL]
#
# data_electrode1 = units_by_multitrodes[MULTITRODE_CHANNEL][0]
#
# pca_ = PCA(n_components=3)
# pca_electrode1 = pca_.fit_transform(data_electrode1)
# isosplit_labels = np.loadtxt(f"./isosplit/realdata_isosplit.csv", delimiter=',', dtype=int)
#
# sp.plot(f'ISO-SPLIT on Electrode 1', pca_electrode1, isosplit_labels, marker='o', alpha=0.5)
# plt.show()
#
# print(ss_metric(labels, isosplit_labels))
# get_metrics(labels, isosplit_labels)