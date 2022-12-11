import os

import numpy as np
from sklearn.decomposition import PCA

import dataset_parsing.simulations_dataset as sds
import dataset_parsing.clustering_datasets as cds

os.chdir("../")


# for SIM_NR in [1, 4, 21, 22, 30]:
#     data, y = sds.get_dataset_simulation_pca_2d(SIM_NR)
#     np.savetxt(f"matlab_sim{SIM_NR}.csv", data)


# data, y = cds.generate_simulated_data()
# np.savetxt(f"matlab_uo.csv", data)




# for nr_comp in [2,3,4,5,6]:
#     data, y = sds.get_dataset_simulation(4)
#     pca_2d = PCA(n_components=nr_comp)
#     spikes_pca_2d = pca_2d.fit_transform(data)
#     np.savetxt(f"matlab_sim{4}_pca{nr_comp}d.csv", spikes_pca_2d)


# for i in range(1, 10):
#     size = i * 250
#     X, y = cds.generate_simulated_data(size)
#     print(len(X))
#     np.savetxt(f"matlab_uo_{len(X)}.csv", X)