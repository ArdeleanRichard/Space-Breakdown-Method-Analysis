import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

from algorithms import ISBM
import dataset_parsing.read_tins_m_data as ds
from validation.comparison import try_metric, compare_metrics_graph_vs_array_structure, \
    compare_result_graph_vs_array_structure, compare_plots_and_metrics
from visualization import scatter_plot as sp
import dataset_parsing.simulations_dataset as sds

import numpy as np
from sklearn.decomposition import PCA

from algorithms import SBM, ISBM
from dataset_parsing.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, plot_multitrode
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel
from visualization.scatter_plot_additionals import plot_spikes_by_clusters


def run_ISBM_graph_on_simulated_data():
    data, y = sds.get_dataset_simulation_pca_2d(4)
    pn = 10
    labels = ISBM.run(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=10) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

    pn = 25
    labels = ISBM.run(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=25) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

    plt.show()



def run_ISBM_graph_on_real_data():
    units_in_channel, labels = ds.get_tins_data()

    # for (i, pn) in list([(4, 25), (6, 40), (17, 15), (26, 30)]):
    for (i, pn) in list([(6, 40)]):
        print(i)
        data = units_in_channel[i-1]
        data = np.array(data)
        pca_2d = PCA(n_components=2)
        X = pca_2d.fit_transform(data)
        km_labels = labels[i-1]

        sp.plot('Ground truth', X, km_labels, marker='o')

        sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
        sp.plot_grid(f'ISBM on Channel {i}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)

        plot_spikes_by_clusters(data, sbm_graph_labels)

        km = KMeans(n_clusters=5).fit(X)
        sp.plot(f'K-means on Channel {i}', X, km.labels_, marker='o')

        plot_spikes_by_clusters(data,  km.labels_)

    plt.show()

def run_ISBM_graph_on_real_data_tetrode():
    DATASET_PATH = '../DATA/TINS/M017_Tetrode/ssd/'

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


    # plot_multitrode(units_by_multitrodes, labels, MULTITRODE_CHANNEL, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
    labels = labels[MULTITRODE_CHANNEL]

    data_electrode1 = units_by_multitrodes[MULTITRODE_CHANNEL][0]

    data_electrode2 = units_by_multitrodes[MULTITRODE_CHANNEL][1]
    data_electrode3 = units_by_multitrodes[MULTITRODE_CHANNEL][2]
    data_electrode4 = units_by_multitrodes[MULTITRODE_CHANNEL][3]
    multitrode = np.hstack([data_electrode1, data_electrode2, data_electrode3, data_electrode4])


    # pca_ = PCA(n_components=8)
    # pca_electrode1 = pca_.fit_transform(multitrode)
    pca_ = PCA(n_components=3)
    pca_vis = pca_.fit_transform(multitrode)

    # compare_plots_and_metrics("M017 multitrode", pca_electrode1, labels, n_clusters=4, eps=150, pn=20, pn2=5, Xvis=pca_vis)

    pca_ = PCA(n_components=3)
    pca_electrode1 = pca_.fit_transform(data_electrode1)
    np.savetxt(f"matlab_realdata.csv", pca_electrode1)
    # compare_plots_and_metrics("Electrode 1", pca_electrode1, labels, n_clusters=4, eps=18, pn=20, pn2=10, Xvis=None)
    compare_plots_and_metrics("Tetrode", pca_vis, labels, n_clusters=4, eps=18, pn=20, pn2=10, Xvis=pca_vis)

    # pca_ = PCA(n_components=2)
    # pca_electrode1 = pca_.fit_transform(data_electrode1)
    # pca_ = PCA(n_components=2)
    # pca_electrode2 = pca_.fit_transform(data_electrode2)
    # pca_ = PCA(n_components=2)
    # pca_electrode3 = pca_.fit_transform(data_electrode3)
    # pca_ = PCA(n_components=2)
    # pca_electrode4 = pca_.fit_transform(data_electrode4)
    # pca_multitrode = np.hstack([pca_electrode1, pca_electrode2, pca_electrode3, pca_electrode4])
    # compare_plots_and_metrics("Electrode 1", pca_multitrode, labels, n_clusters=4, eps=18, pn=20, pn2=10, Xvis=pca_vis)






if __name__ == '__main__':
    run_ISBM_graph_on_simulated_data()
    # run_ISBM_graph_on_real_data()
    # run_ISBM_graph_on_real_data_tetrode()

