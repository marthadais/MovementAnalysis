# This file is part of MovementAnalysis.
#
# [1] Ferreira, M. D., Campbell, J. N., & Matwin, S. (2022).
# A novel machine learning approach to analyzing geospatial vessel patterns using AIS data.
# GIScience & Remote Sensing, 59(1), 1473-1490.
#
from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models
from approach.clustering import Clustering
from datetime import datetime
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dist_matrix_plot(dm_path, path='./results/'):
    """
    It generates heatmap for the distance matrices.

    :param dm_path: path where the distance matrix is saved
    :param path: path to save the images
    """
    if not os.path.exists(f'{path}features'):
        os.mkdir(f'{path}features')

    for f in dm_path.keys():
        dm = pickle.load(open(dm_path[f], 'rb'))
        # change the fontsize of the xtick and ytick labels and axes
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.imshow(dm, cmap='Blues_r', interpolation='nearest')
        plt.xlabel('Number of instances', fontsize=15)
        plt.ylabel('Number of instances', fontsize=15)
        plt.colorbar()
        plt.savefig(f'{path}features/{f}_dist_matrix.png', bbox_inches='tight')
        plt.close()

        dm_log = np.log(dm + 1e-10)
        plt.imshow(dm_log, cmap='Blues_r', interpolation='nearest')
        plt.xlabel('Number of instances')
        plt.ylabel('Number of instances')
        plt.colorbar()
        plt.savefig(f'{path}features/{f}_log_dist_matrix.png', bbox_inches='tight')
        plt.close()


def all_clustering(dataset, features_path, folder, path_results_dict, metric, eps=None, k1=None, k2=None, norm_dist=False):
    """
    It executes all the three clustering algorithms for a given dataset.

    :param dataset: path to file that contains the preprocessed dataset
    :param features_path: path to file that contains the coefficients produced for each trajectory in the dataset
    :param folder: folder path to save the results
    :param path_results_dict: dict to save the path of the results
    :param metric: regression algorithm used to produce the coefficients
    :param eps: epsilon for the DBSCAN algorithm (Default: None).
    :param k1: number of clusters for HC algorithm (Default: None).
    :param k2: number of clusters for SC algorithm (Default: None).
    :param norm_dist: if True, it normalizes the distance matrix (Default: False).
    :return: dict with the path of the results
    """
    ## Clustering
    if eps is None:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                        cluster_algorithm='dbscan', folder=folder, norm_dist=norm_dist)
    else:
        result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                            cluster_algorithm='dbscan', folder=folder, eps=eps, norm_dist=norm_dist)
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

# dict with path of the results
path_results_dict = {}
path_features_dict = {}
path_dist_dict = {}

print('Starting all Experiments...')
n_samples = None
# Fishing
# https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
vessel_type = [30, 1001, 1002]
# Dates
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']

# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))
main_folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/mov-{dim_set}-{n_samples}-'

# Extracting features using OU
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
# clustering using OU
path_results_dict = all_clustering(dataset, distance_path, folder, path_results_dict, metric=metric, eps=0.02, k1=5, k2=3)

# Extracting features using ARIMA
print('Running ARIMA process...')
ar_arima = 1
i_arima = 0
ma_arima = 3
metric = 'arima'
folder = f'{main_folder}{metric}-{ar_arima}-{i_arima}-{ma_arima}/'

features_path = f'{folder}/features_coeffs.csv'
distance_path = f'{folder}/features_distance.p'
path_features_dict['arima-features'] = features_path
path_dist_dict['arima-dm'] = distance_path
if not os.path.exists(features_path):
    dataset_dict = dataset.pandas_to_dict()
    features = Models(dataset=dataset_dict, features_opt=metric, dim_set=dim_set, ar_prm=ar_arima, i_prm=i_arima, ma_prm=ma_arima, folder=folder)
# clustering using ARIMA
path_results_dict = all_clustering(dataset, distance_path, folder, path_results_dict, metric=metric, eps=0.1, k1=5, k2=3)

# print paths with results
print(path_results_dict)
print(path_results_dict.keys())

# compute plots of the distances
dist_matrix_plot(path_dist_dict, main_folder)

# Compute measure between clustering algorithms
lbl = pd.DataFrame()
for exp in path_results_dict.keys():
    data = pd.read_csv(path_results_dict[exp])
    lbl[exp] = data.groupby('mmsi').first()['Clusters']

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
mat1 = pd.DataFrame(columns=path_results_dict.keys(), index=path_results_dict.keys())
mat2 = pd.DataFrame(columns=path_results_dict.keys(), index=path_results_dict.keys())
for exp1 in path_results_dict.keys():
    for exp2 in path_results_dict.keys():
        mat1.loc[exp1, exp2] = normalized_mutual_info_score(lbl[exp1], lbl[exp2])
        mat2.loc[exp1, exp2] = adjusted_rand_score(lbl[exp1], lbl[exp2])


out = pd.DataFrame()
for exp in path_results_dict.keys():
    data = pd.read_csv(path_results_dict[exp])
    out[exp] = data.groupby('mmsi').first()['scores-3std']


