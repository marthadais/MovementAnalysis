# This file is part of MovementAnalysis.
#
# [1] Ferreira, M. D., Campbell, J. N., & Matwin, S. (2022).
# A novel machine learning approach to analyzing geospatial vessel patterns using AIS data.
# GIScience & Remote Sensing, 59(1), 1473-1490.
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
import os, pickle
from joblib import Parallel, delayed
from analysis import projection as pjt
from approach import OU_process as ou
from sklearn.preprocessing import MinMaxScaler


def dict_reorder(x):
    """
    It reorder the dict values.

    :param x: the data on dict format
    :return: dict ordered
    """
    return {k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(x.items())}


def avg_std_dict_data(x, dim_set):
    """
    It computes the average and the standard deviation of a dict dataset considering a set of attributes.

    :param x: the dataset in dict format
    :param dim_set: a list of the attributes to be computed
    :return: average and standard deviation
    """
    avg = {}
    std = {}
    maxv = {}
    for dim in dim_set:
        aux = np.concatenate([x[k].get(dim) for k in x])
        avg[dim] = aux.mean()
        std[dim] = aux.std()
        maxv[dim] = aux.max()
    return avg, std, maxv


def normalize(x, dim_set, verbose=True, znorm=True, centralize=False, norm_geo=True):
    """
    Computes Z-normalization or centralization of a dict dataset for a set of attributes.

    :param x: dict dataset
    :param dim_set: set of attributes
    :param verbose: if True, print comments
    :param znorm: if True, it computes the z-normalization
    :param centralize: it True, it computes the centralization
    :return: normalized dict dataset
    """
    if verbose:
        print(f"Normalizing dataset")
    avg, std, maxv = avg_std_dict_data(x, dim_set)

    ids = list(x.keys())
    for id_a in range(len(ids)):
        # normalize features
        if znorm:
            for dim in dim_set:
                x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]
        elif centralize:
            for dim in dim_set:
                x[ids[id_a]][dim] = x[ids[id_a]][dim]-avg[dim]
        elif norm_geo:
            for dim in dim_set:
                if (dim == 'lat') or (dim == 'LAT'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/90
                elif (dim == 'lon') or (dim == 'LON'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/180
                else:
                    # x[ids[id_a]][dim] = x[ids[id_a]][dim]/maxv[dim]
                    x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]

    return x


class Models:
    """
    It reads the preprocessed DCAIS dataset in dict format.
    Next, the selected model is applied to extract the coefficients that represents the movement of the vessel trajectory.
    """
    def __init__(self, dataset, verbose=True, **args):
        """
        It receives the preprocessed DCAIS dataset in dict format.
        It applies the selected model on the trajectories and compute the euclidean distance.

        :param dataset: the dataset in dict format
        """
        self.verbose = verbose
        self.dm = None
        self.coeffs = None
        self.measures = None
        # self.num_cores = 2*(multiprocessing.cpu_count()//3)
        self.num_cores = 3
        if 'njobs' in args.keys():
            self.num_cores = args['njobs']

        self.features_opt = 'arima'
        if 'features_opt' in args.keys():
            self.features_opt = args['features_opt']

        self._dim_set = ['lat', 'lon']
        if 'dim_set' in args.keys():
            self._dim_set = args['dim_set']

        self._znorm = False
        if 'znorm' in args.keys():
            self._znorm = args['znorm']

        self._centralize = False
        if 'centralize' in args.keys():
            self._centralize = args['centralize']

        self._normalizationGeo = True
        if 'norm_geo' in args.keys():
            self._normalizationGeo = args['norm_geo']

        self.dataset_norm = dataset
        if self._znorm or self._centralize or self._normalizationGeo:
            self.dataset_norm = normalize(self.dataset_norm, self._dim_set, verbose=verbose, znorm=self._znorm, centralize=self._centralize, norm_geo=self._normalizationGeo)
        self._ids = list(self.dataset_norm.keys())

        # methods parameters
        self.ar_prm = 1
        if 'ar_prm' in args.keys():
            self.ar_prm = args['ar_prm']

        self.ma_prm = 1
        if 'ma_prm' in args.keys():
            self.ma_prm = args['ma_prm']

        self.i_prm = 0
        if 'i_prm' in args.keys():
            self.i_prm = args['i_prm']

        _metrics_dict = self.create_data_dict()
        _metrics_dict[self.features_opt]()

        # saving features
        if 'folder' in args.keys():
            self.path = args['folder']

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            df_features = pd.DataFrame(self.coeffs)
            df_features.to_csv(f'{self.path}/features_coeffs.csv')
            if self.features_opt in ['arima', 'arima_multi']:
                self.measures.to_csv(f'{self.path}/features_measures.csv')
            pjt.plot_coeffs_traj(self.coeffs, pd.Series(np.zeros(self.coeffs.shape[0])), folder=self.path)

            pickle.dump(self.dm, open(f'{self.path}/features_distance.p', 'wb'))
            df_features = pd.DataFrame(self.dm)
            df_features.to_csv(f'{self.path}/features_distance.csv')
            self.dm_path = f'{self.path}/features_distance.p'

    def arima_coefs(self):
        """
        It computes ARIMA model in each trajectory, and produce the distance matrix with Euclidean distance.
        """
        measure_pd = {}
        coeffs = {}
        for i in range(len(self._ids)):
            measure_pd[self._ids[i]] = {}
            coeffs[self._ids[i]] = {}

        Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._arima_func)(i, measure_pd, coeffs) for i in list(range(len(self._ids))))

        measure_pd = pd.DataFrame(measure_pd).T
        col_names_all = ['AIC', 'BIC', 'MSE', 'MAE', 'SSE', 'HQIC']
        col_names = ['mmsi']
        for dim in range(len(self._dim_set)):
            col_names = col_names + col_names_all
        measure_pd.columns = col_names
        self.measures = measure_pd

        coeffs = dict_reorder(coeffs)
        self.coeffs = np.array([coeffs[item] for item in coeffs.keys()])
        self.coeffs[np.isnan(self.coeffs)] = 0
        scaler = MinMaxScaler().fit(self.coeffs)
        self.coeffs = scaler.transform(self.coeffs)
        self.dm = squareform(pdist(self.coeffs))
        self.dm = self.dm / self.dm.max()

    def ou(self):
        """
        It computes OU model in each trajectory, and produce the distance matrix with Euclidean distance.
        """
        coeffs = {}
        for i in range(len(self._ids)):
            coeffs[self._ids[i]] = {}
        Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._ou_func)(i, coeffs) for i in list(range(len(self._ids))))
        coeffs = dict_reorder(coeffs)
        self.coeffs = np.array([coeffs[item] for item in coeffs.keys()])
        scaler = MinMaxScaler()
        scaler.fit(self.coeffs)
        self.coeffs = scaler.transform(self.coeffs)
        self.dm = squareform(pdist(self.coeffs))
        self.dm = self.dm / self.dm.max()

    ### functions to parallelize ###
    def _arima_func(self, i, measure_pd, coeffs):
        """
        It computes ARIMA model for one trajectory.
        """
        coeffs_i = np.array([])
        measure_list = [self.dataset_norm[self._ids[i]]['mmsi'][0]]
        if self.verbose:
            print(f"Computing {i} of {len(self._ids)}")
        for dim in self._dim_set:
            st = self.dataset_norm[self._ids[i]][dim]
            model = sm.tsa.SARIMAX(st, order=(self.ar_prm, self.i_prm, self.ma_prm), trend='c',
                                   enforce_stationarity=False)
            res = model.fit(disp=False)
            coeffs_i = np.hstack((coeffs_i, res.params))
            measure_list = measure_list + [res.aic, res.bic, res.mse, res.mae, res.sse, res.hqic]

        measure_pd[self._ids[i]] = measure_list
        coeffs[self._ids[i]] = coeffs_i

    def _ou_func(self, i, coeffs):
        """
        It computes OU model for one trajectory.
        """
        coeffs_i = np.array([])
        if self.verbose:
            print(f"Computing {i} of {len(self._ids)}")
        st_time = self.dataset_norm[self._ids[i]]['time'].astype('datetime64[s]')
        st_time = np.hstack((0, np.diff(st_time).cumsum().astype('float')))

        for dim in self._dim_set:
            st = self.dataset_norm[self._ids[i]][dim]
            st = st.reshape((1, len(st)))
            res = ou.ou_process(st_time, st)
            coeffs_i = np.hstack((coeffs_i, res))

        coeffs[i] = coeffs_i

    def create_data_dict(self):
        """
        Dictionary of models options.
        """
        return {'ou': self.ou,
                'arima': self.arima_coefs}


