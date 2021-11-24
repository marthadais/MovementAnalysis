from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models
from approach.clustering import Clustering
from datetime import datetime
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import projection as pjt


def dist_matrix_plot(features, dm_path, path='./results/'):
    """
    It generate heatmap for the distance matrices and the PCA projection.
    :param features: coefficients of the AR models
    :param dm_path: path where the distance matrix is saved
    :param path: path to save the images
    """
    if not os.path.exists(f'{path}features'):
        os.mkdir(f'{path}features')

    for f in dm_path.keys():
        dm = pickle.load(open(dm_path[f], 'rb'))
        plt.imshow(dm, cmap='viridis', interpolation='nearest')
        plt.xlabel('Number of instances')
        plt.ylabel('Number of instances')
        plt.colorbar()
        plt.savefig(f'{path}features/{f}_dist_matrix.png', bbox_inches='tight')
        plt.close()

        dm_log = np.log(dm + 1e-10)
        plt.imshow(dm_log, cmap='viridis', interpolation='nearest')
        plt.xlabel('Number of instances')
        plt.ylabel('Number of instances')
        plt.colorbar()
        plt.savefig(f'{path}features/{f}_log_dist_matrix.png', bbox_inches='tight')
        plt.close()

    for f in features.keys():
        x = pd.read_csv(features[f], index_col=[0])
        pjt.plot_traj_pca(x, f'{path}features/{f}')


def all_clustering(dataset, features_path, folder, path_results_dict, metric, eps=None, k1=None, k2=None, norm_dist=False):
    ### Clustering
    if eps is None:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                        cluster_algorithm='dbscan', folder=folder)
    else:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                            cluster_algorithm='dbscan', folder=folder, eps=eps)
    path_results_dict[f'{metric}-dbscan'] = result.results_file_path

    if k1 is None:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                        cluster_algorithm='hierarchical', linkage='average', folder=folder, norm_dist=norm_dist)
    else:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                            cluster_algorithm='hierarchical', linkage='average', folder=folder, k=k1, norm_dist=norm_dist)
    path_results_dict[f'{metric}-average'] = result.results_file_path

    if k2 is None:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                        cluster_algorithm='spectral', folder=folder, norm_dist=norm_dist)
    else:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                            cluster_algorithm='spectral', folder=folder, k=k2, norm_dist=norm_dist)
    path_results_dict[f'{metric}-spectral'] = result.results_file_path

    return path_results_dict

path_results_dict = {}
path_features_dict = {}
path_dist_dict = {}

print('Starting all Experiments...')
n_samples = None
## Fishing
vessel_type = [30, 1001, 1002]
## Dates
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']

### Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))
main_folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/mov-{n_samples}-'

#### Extracting features
print('Running OU process...')
metric = 'ou'
folder = f'{main_folder}{metric}/'

features_path = f'{folder}/features_coeffs.csv'
distance_path = f'{folder}/features_distance.p'
path_features_dict['ou-features'] = features_path
path_dist_dict['ou-dm'] = distance_path
if not os.path.exists(features_path):
    dataset_dict = dataset.pandas_to_dict()
    features = Models(dataset=dataset_dict, features_opt=metric, dim_set=dim_set, folder=folder)

path_results_dict = all_clustering(dataset, distance_path, folder, path_results_dict, metric=metric)

print('Running ARIMA process...')
ar_arima = 1
i_arima = 0
ma_arima = 2
metric = 'arima'
folder = f'{main_folder}{metric}-{ar_arima}-{i_arima}-{ma_arima}/'

features_path = f'{folder}/features_coeffs.csv'
distance_path = f'{folder}/features_distance.p'
path_features_dict['arima-features'] = features_path
path_dist_dict['arima-dm'] = distance_path
if not os.path.exists(features_path):
    dataset_dict = dataset.pandas_to_dict()
    features = Models(dataset=dataset_dict, features_opt=metric, dim_set=dim_set, ar_prm=ar_arima, i_prm=i_arima, ma_prm=ma_arima, folder=folder)

path_results_dict = all_clustering(dataset, distance_path, folder, path_results_dict, metric=metric)

print(path_results_dict)
print(path_results_dict.keys())

dist_matrix_plot(path_features_dict, path_dist_dict, main_folder)
