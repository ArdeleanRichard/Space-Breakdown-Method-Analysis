# Space Breakdown Method Analysis
Space Breakdown Method (SBM) is a clustering algorithm that can be used to cluster low-dimensional neural data with efficiency, due to its linear complexity scaling with the data size.
SBM has been published by IEEE in September 2019:
- DOI: 10.1109/ICCP48234.2019.8959795
- Conference: 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP)


## Git repository structure
This repository is the complete version of the 'Space-Breakdown-Method' repository including the analysis of various algorithms in comparison with SBM. 

The code folder structure:
- run.py : main file to run the code from
- requirements.txt : python libraries that are needed in order to run the code
- images: images that will be used for this readme
- algorithms : folder that contains the actual SBM and ISBM code
- dataset parsing: scripts for parsing different datasets
- common/validation/visualization: utilitary functions

## Setup
The 'requirements.txt' file indicates the dependencies required for running the code. The data used in this study can be downloaded from: https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO. 

The paths to the data folder on your local workstation need to be set from the 'constants.py' file (DATA_FOLDER_PATH, SIM_DATA_FOLDER_PATH, REAL_DATA_FOLDER_PATH).


## Citation
We would appreciate it if you cite the paper when you use this work:

- For Plain Text:
```
E. Ardelean, A. Stanciu, M. Dinsoreanu, R. Potolea, C. Lemnaru and V. V. Moca, "Space Breakdown Method A new approach for density-based clustering," 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP), 2019, pp. 419-425, doi: 10.1109/ICCP48234.2019.8959795.
```

- BibTex:
```
@INPROCEEDINGS{8959795,
  author={Ardelean, Eugen-Richard and Stanciu, Alexander and Dinsoreanu, Mihaela and Potolea, Rodica and Lemnaru, Camelia and Moca, Vasile Vlad},
  booktitle={2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP)}, 
  title={Space Breakdown Method A new approach for density-based clustering}, 
  year={2019},
  volume={},
  number={},
  pages={419-425},
  doi={10.1109/ICCP48234.2019.8959795}}
```


# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)