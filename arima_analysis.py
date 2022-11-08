# This file is part of MovementAnalysis.
#
# [1] Ferreira, M. D., Campbell, J. N., & Matwin, S. (2022).
# A novel machine learning approach to analyzing geospatial vessel patterns using AIS data.
# GIScience & Remote Sensing, 59(1), 1473-1490.
#
from preprocessing.clean_trajectories import Trajectories
from approach.ar_models import Models
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def boxplot_measure(data, measure, folder, measure2=None, limit=None):
    """
    It plots the box plots for all 24 configurations of latitude and longitude arima models.

    :param data: the measurements of the ARIMA models.
    :param measure: the selected measure for the first plot (latitude).
    :param folder: folder to save the images
    :param measure2: the selected measure for the second plot (longitude).
    :param limit: limits for y-axis in the plot.
    """
    x = pd.DataFrame()
    if measure2 is not None:
        x1 = pd.DataFrame()
    for i in data.keys():
        x = pd.concat([x, data[i][measure]], axis=1)
        if measure2 is not None:
            x1 = pd.concat([x1, curr_config[i][f'{measure}.1']], axis=1)

    if measure == 'MSE':
        x = np.sqrt(x)
        if measure2 is not None:
            x1 = np.sqrt(x1)

    x.columns = data.keys()
    x = x.replace([np.inf], np.nan)
    x = x.replace([np.nan], x.max().max())
    if measure2 is not None:
        x1.columns = data.keys()
        x1 = x1.replace([np.inf], np.nan)
        x1 = x1.replace([np.nan], x1.max().max())

    # fig = plt.figure(figsize=(15, 10))
    print(f'{measure}:')
    print(f'{pd.concat([x.mean(), x.std()], axis=1)}')
    if measure2 is not None:
        print(f'{pd.concat([x1.mean(), x1.std()], axis=1)}')

    # Plot
    col_names = [i.replace('_', ',') for i in x.columns]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    bp1 = ax.boxplot(x)

    # change the fontsize of the xtick and ytick labels and axes
    ax.set_ylim([limit[0], limit[1]])
    ax.set_xticklabels(col_names, rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)

    plt.savefig(f'{folder}/features_arima_latitude_{measure}.png', bbox_inches='tight')
    plt.close()

    if measure2 is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        bp2 = ax.boxplot(x1)

        # change the fontsize of the xtick and ytick labels and axes
        ax.set_ylim([limit[0], limit[1]])
        ax.set_xticklabels(col_names, rotation=45, fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_ylabel('MSE', fontsize=15)

        plt.savefig(f'{folder}/features_arima_longitude_{measure}.png', bbox_inches='tight')
        plt.close()

    all_stats = get_box_plot_data(x.columns, bp1, x)
    if measure2 is not None:
        lon_stats = get_box_plot_data(x.columns, bp2, x1)
        all_stats = pd.concat([all_stats, lon_stats], axis=0)
    all_stats.to_csv(f'{folder}/{measure}_stats.csv')
    decimals = 4

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
        avg = round(row1['average'], decimals)
        st = round(row1['std'], decimals)
        avg2 = round(row2['average'], decimals)
        st2 = round(row2['std'], decimals)
        avg3 = round(row3['average'], decimals)
        st3 = round(row3['std'], decimals)
        print(f'$({lbl})$ & ${avg} \pm {st}$ & ${avg2} \pm {st2}$ & ${avg3} \pm {st3}$ \\\\')


def get_box_plot_data(labels, bp, data):
    """
    It gets the boxplot and data information and save into a csv file.

    :param labels: the configuration under analysis
    :param bp: the boxplot information
    :param data: the results of the configuration under analysis.

    :return: a summary of all the boxplot and data information
    """
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
n_samples = 300
## Fishing
vessel_type = [30, 1001, 1002]
## Dates
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)

dataset = Trajectories(dataset='DCAIS', n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day))
main_folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/arima_analysis_sog_{n_samples}'

metric = 'arima'
dim_set = ['lat', 'lon']

curr_config = {}
silhouette = pd.DataFrame()
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


folder = f'{main_folder}/'
boxplot_measure(curr_config, 'MSE', measure2='MSE', limit=(-0.01, 0.1), folder=folder)

