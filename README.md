# Movement Analysis of Fishing vessels

An analysis on the behavior of fishing vessels in order to detect patterns and outliers/anomalies.
This analysis utilizes ARIMA and OU process to model the trajectories, producing coefficients that represent the vessel movement.
Next, a clustering analysis is performed to explore patterns in the movement of the fishing vessels.

## Requirements

## Usage Example

This is an example of how to run the movement analysis on DCAIS dataset.
In this scenario, we apply OU process in 30 trajectories of fishing vessels that was navigating on April, 2020.
The clustering algorithm executed is the hierarchical with average-linkage metric.

Import the requires libraries:
```python
from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models
from approach.clustering import Clustering
from datetime import datetime
```

Process the dataset online:
```python
# Number of vessels
n_samples = 30
# Fishing type
vessel_type = [30, 1001, 1002]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']
# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))
```

Modeling the trajectories and apply the clustering algorithm:
```python
main_folder = f'./results/DCAIS_example/'
#### Extracting features
dataset_dict = dataset.pandas_to_dict()
features = Models(dataset=dataset_dict, features_opt='ou', dim_set=dim_set, folder=f'./results/DCAIS_example/')
### Runing clustering
result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=f'./results/DCAIS_example/features_coeffs.csv',
                        cluster_algorithm='hierarchical', linkage='average', folder=f'./results/DCAIS_example/', norm_dist=False)
```

## Files description