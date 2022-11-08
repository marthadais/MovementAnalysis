# This file is part of MovementAnalysis.
#
# [1] Ferreira, M. D., Campbell, J. N., & Matwin, S. (2022).
# A novel machine learning approach to analyzing geospatial vessel patterns using AIS data.
# GIScience & Remote Sensing, 59(1), 1473-1490.
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def sc_bar_plt(data, folder='./results_data/'):
    """
    It generates the silhouette score bar graph.

    :param data: the dataset
    :param folder: the path were is the clustering results to save the statistics
    """
    lower_limit = data.loc[0, 'threshold_std']
    avg_limit = data['silhouette'].mean()
    data = data.sort_values([data.columns[1], data.columns[0]])
    data = data.reset_index()[[data.columns[0], data.columns[1]]]

    color_order = ['red', 'orange', 'blue', 'green', 'yellow', 'pink', 'violet', 'maroon', 'wheat', 'yellowgreen',
                   'lime', 'indigo', 'azure', 'olive', 'cyan', 'beige', 'skyblue', 'lavender', 'gold', 'fuchsia',
                   'purple']

    pad = 0
    i = 1
    n_clusters = len(np.unique(data.iloc[:, 1]))
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111)
    for c in range(n_clusters):
        if np.unique(data.iloc[:, 1])[c] == -1:
            i = 0
        if n_clusters <= 20:
            curr_c = color_order[i]
        else:
            curr_c = cm.tab20(float(c) / n_clusters)
        sample = data[data.iloc[:, 1] == np.unique(data.iloc[:, 1])[c]]
        ax.barh(range(pad, len(sample.iloc[:, 0])+pad), sample.iloc[:, 0], label=f'{np.unique(data.iloc[:, 1])[c]}', color=curr_c)
        pad = pad + len(sample.iloc[:, 0]) + 2
        i = i+1

    ax.plot([lower_limit, lower_limit], [0, pad-1], "--", label=f'lower_limit', color='red')
    ax.plot([avg_limit, avg_limit], [0, pad-1], "--", label=f'average', color='black')
    ax.set_ylabel('Instances', fontsize=25)
    ax.set_xlabel('Silhouette Score', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # ax.set_title(f'Individual Silhouette Score for {n_clusters} groups')
    if n_clusters < 20:
        plt.legend(fontsize=25)
    # plt.tight_layout()
    plt.savefig(f'{folder}/silhoutte.png', bbox_inches='tight')
    plt.close()


def statistics(dataset, col='trajectory', folder='./results_data/'):
    """
    It computes the statistics based on one attribute.

    :param dataset: the dataset
    :param col: the attribute to compute statistics (Default: 'trajectory')
    :param folder: the path were is the clustering results to save the statistics
    """
    vt_trajectory = dataset[col].unique()
    id_statistcs = pd.DataFrame()
    for i in vt_trajectory:
        sample = dataset[dataset[col] == i]
        row = [i, sample['mmsi'].iloc[0], sample['Clusters'].iloc[0], sample.shape[0],
               sample['silhouette'].iloc[0], sample['scores-3std'].iloc[0]]
        row = pd.DataFrame([row], columns=[col, 'mmsi', 'cluster', 'n_observations',
                                           'silhouette', 'scores'])
        id_statistcs = pd.concat([id_statistcs, row], ignore_index=True)

    id_statistcs.to_csv(f'{folder}/trajectory_statistcs_measure.csv')


def file_statistics(file, directory):
    """
    It receives the path with the clustering results and the folder path to compute the statistics and plot images.

    :param file: the path were is the dataset.
    :param directory: the path were is the clustering results to save the statistics.
    """
    dataset = pd.read_csv(file)
    dataset['time'] = dataset['time'].astype('datetime64[ns]')

    trajectory_df = dataset.loc[:, ['trajectory', 'silhouette', 'Clusters', 'threshold_std']]
    trajectory_df.drop_duplicates(inplace=True)
    sc_bar_plt(trajectory_df[['silhouette', 'Clusters', 'threshold_std']], folder=directory)

    print('\t summarizing trajectories info')
    statistics(dataset, folder=directory)


