from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def boxplot_measure(data, measure, folder, measure2=None, measure3=None, limit=None):
    x = pd.DataFrame()
    if measure2 is not None:
        x1 = pd.DataFrame()
        x2 = pd.DataFrame()
    for i in data.keys():
        x = pd.concat([x, data[i][measure]], axis=1)
        if measure2 is not None:
            x1 = pd.concat([x1, curr_config[i][f'{measure}.1']], axis=1)
        if measure3 is not None:
            x2 = pd.concat([x2, curr_config[i][f'{measure}.2']], axis=1)

    if measure == 'MSE':
        x = np.sqrt(x)
        if measure2 is not None:
            x1 = np.sqrt(x1)
        if measure3 is not None:
            x2 = np.sqrt(x2)

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    # x[x >= 140] = np.nan
    x = x.replace([np.nan], x.max().max())
    if measure2 is not None:
        x1.columns = data.keys()
        x1 = x1.replace([np.inf], np.nan)
        # x1[x1 >= 140] = np.nan
        x1 = x1.replace([np.nan], x1.max().max())
    if measure3 is not None:
        x2.columns = data.keys()
        x2 = x2.replace([np.inf], np.nan)
        # x1[x1 >= 140] = np.nan
        x2 = x2.replace([np.nan], x2.max().max())

    # fig = plt.figure(figsize=(15, 10))
    print(f'{measure}:')
    print(f'{pd.concat([x.mean(), x.std()], axis=1)}')
    if measure2 is not None:
        print(f'{pd.concat([x1.mean(), x1.std()], axis=1)}')
    if measure3 is not None:
        print(f'{pd.concat([x2.mean(), x2.std()], axis=1)}')
    # print(aic1.mean())
    if measure2 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    else:
        if measure3 is None:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        ax = axes[0]

    if limit is not None:
        ax.set_ylim([limit[0], limit[1]])

    ax.title.set_text('Latitute')
    ax.set_xticklabels(x.columns)
    # bp1 = x.boxplot(ax=axes[0])
    bp1 = ax.boxplot(x)
    if measure2 is not None:
        if limit is not None:
            axes[1].set_ylim([limit[0], limit[1]])
        axes[1].title.set_text('Longitude')
        axes[1].set_xticklabels(x.columns)
        # bp2 = x1.boxplot(ax=axes[1])
        bp2 = axes[1].boxplot(x1)
    if measure3 is not None:
        if limit is not None:
            axes[2].set_ylim([limit[0], limit[1]])
        axes[2].title.set_text('SOG')
        axes[2].set_xticklabels(x.columns)
        # bp2 = x1.boxplot(ax=axes[1])
        bp3 = axes[2].boxplot(x2)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{folder}/features_arima_{measure}.png', bbox_inches='tight')
    plt.close()

    all_stats = get_box_plot_data(x.columns, bp1, x)
    if measure2 is not None:
        lon_stats = get_box_plot_data(x.columns, bp2, x1)
        all_stats = pd.concat([all_stats, lon_stats], axis=0)
    if measure3 is not None:
        sog_stats = get_box_plot_data(x.columns, bp3, x2)
        all_stats = pd.concat([all_stats, sog_stats], axis=0)
    all_stats.to_csv(f'{folder}/{measure}_stats.csv')

    for index in range(24):
        row1 = all_stats.iloc[index,:]
        row2 = all_stats.iloc[index+24,:]
        row3 = all_stats.iloc[index+48,:]
        lbl = row1['label']
        lbl = lbl.replace('_', ',')
        lbl2 = row2['label']
        lbl2 = lbl2.replace('_', ',')
        lbl3 = row3['label']
        lbl3 = lbl3.replace('_', ',')
        avg = round(row1['average'], 2)
        st = round(row1['std'], 2)
        avg2 = round(row2['average'], 2)
        st2 = round(row2['std'], 2)
        avg3 = round(row3['average'], 2)
        st3 = round(row3['std'], 2)
        print(f'$({lbl})$ & ${avg} \pm {st}$ & ${lbl2}$ & ${avg2} \pm {st2}$ & ${lbl3}$ & ${avg3} \pm {st3}$ \\\\')


def get_box_plot_data(labels, bp, data):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        dict1['average'] = data[labels[i]].mean(axis=0)
        dict1['std'] = data[labels[i]].std(axis=0)
        dict1['n_outliers_lower'] = (data[labels[i]] < dict1['lower_whisker']).sum()
        dict1['n_outliers_lower_prc'] = (data[labels[i]] < dict1['lower_whisker']).sum() / data[labels[i]].shape[0]
        dict1['n_outliers_upper'] = (data[labels[i]] > dict1['upper_whisker']).sum()
        dict1['n_outliers_upper_prc'] = (data[labels[i]] > dict1['upper_whisker']).sum() / data[labels[i]].shape[0]
        dict1['n_between_0_max'] = (data[labels[i]][data[labels[i]] >= 0] <= dict1['upper_whisker']).sum()
        dict1['n_between_0_max_prc'] = (data[labels[i]][data[labels[i]] >= 0] <= dict1['upper_whisker']).sum() / data[labels[i]].shape[0]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


print('Starting all Experiments...')
dataset_name = 'noaa'
n_samples = 300
# n_samples = 30
region_limits = None
# region_limits = [30, 38, -92, -70]

## Tug Tow
# vessel_type = [21, 22, 31, 32, 52, 1023, 1025]

## Fishing
# commercial fishing = 1001
# Fish Processing = 1002
vessel_type = [30, 1001, 1002]

## Dates
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)

dataset = Trajectories(dataset=dataset_name, n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day), region=region_limits)
main_folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/arima_analysis_sog_{n_samples}'
# main_folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/exp-'

metric = 'arima'
dim_set = ['lat', 'lon', 'sog']

curr_config = {}
silhouette = pd.DataFrame()
# n_groups = [4, 6, 6, 2, 2, 2, 2, 2, 2, 3, 2, 8, 2, 7, 2, 2, 2, 2]
count = 0
for ar_i in [0, 1]:
    for ar_p in [1, 2, 3]:
        for ar_q in [0, 1, 2, 3]:
            print(f'{ar_p}_{ar_i}_{ar_q}')
            folder = f'{main_folder}/{metric}_{ar_p}_{ar_i}_{ar_q}/'
            features_path = f'{folder}/features_distance.p'
            if not os.path.exists(features_path):
                dataset_dict = dataset.pandas_to_dict()
                features = Models(dataset=dataset_dict, features_opt=metric, dim_set=dim_set,
                                  ar_prm=ar_p, i_prm=ar_i, ma_prm=ar_q, folder=folder, znorm=False)

            file_path = f'{main_folder}/{metric}_{ar_p}_{ar_i}_{ar_q}/features_measures.csv'
            curr_config[f'{ar_p}_{ar_i}_{ar_q}'] = pd.read_csv(file_path)


col_set = ['AIC', 'MSE']
# col_lim_upper = [2e5, 2e5, 2e5, 100, 1, 2e4, 2e5]
col_lim_upper = [2e4, 15]
# col_lim_lower = [-2e5, -2e5, -2e5, -10, -0.01, -10, -2e5]
col_lim_lower = [-8e4, -1]

folder = f'{main_folder}/'
boxplot_measure(curr_config, 'MSE', measure2='MSE', measure3='MSE', limit=(-1, 20), folder=folder)
# boxplot_measure(curr_config, 'AIC', measure2='AIC', limit=(-2e4, 2e4))
# boxplot_measure(curr_config, 'MSE', limit=(-1, 50))
# boxplot_measure(curr_config, 'AIC', limit=(2e4, -2e4))

