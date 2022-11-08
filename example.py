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

# Number of vessels
n_samples = None
# Fishing type
vessel_type = [30, 1001, 1002]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']
# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))

main_folder = f'./results/DCAIS_example/'
#### Extracting features
dataset_dict = dataset.pandas_to_dict()
features = Models(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder=f'./results/DCAIS_example/')
### Runing clustering
result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=f'./results/DCAIS_example/features_distance.p',
                        cluster_algorithm='hierarchical', linkage='average', folder=f'./results/DCAIS_example/', norm_dist=False)

