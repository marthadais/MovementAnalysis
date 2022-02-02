from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def pca_coeffs(x):
    """
    It applies PCA projection using 2 components.
    :param x: the dataset
    :return: the projection of the dataset in 2d, percentage of PC1, percentage of PC2
    """
    pca_model = PCA(n_components=2)
    pca_model.fit(x)
    traj_2d = pca_model.fit_transform(x)
    return traj_2d, pca_model.explained_variance_ratio_[0]*100, pca_model.explained_variance_ratio_[1]*100


def plot_traj_pca(x, folder):
    """
    It generates the plot with PCA projection for 2 components.
    :param x: the dataset
    :param folder: the folder to save the image
    """
    # PCA
    traj_2d, pc1, pc2 = pca_coeffs(x)
    fig = plt.figure(1)
    # change the fontsize of the xtick and ytick labels and axes
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)
    plt.scatter(traj_2d[:, 0], traj_2d[:, 1], color='gray')
    plt.xlabel(f'PC1 ({round(pc1,2)}%)')
    plt.ylabel(f'PC2 ({round(pc2,2)}%)')
    plt.savefig(f'{folder}_projection_PCA.png', bbox_inches='tight')
    plt.close()


def plot_coeffs_traj(x, clusters, folder):
    """
    It generates the plot with PCA projection for 2 components colored based on the clustering results.
    :param x: the dataset
    :param clusters: the dataset labels
    :param folder: the folder to save the image
    """
    traj_2d, pc1, pc2 = pca_coeffs(x)
    n_cluster = clusters.unique()
    n_cluster.sort()

    color_order = ['red', 'orange', 'blue', 'green', 'yellow', 'pink', 'violet', 'maroon', 'wheat', 'yellowgreen',
                   'lime', 'indigo', 'azure', 'olive', 'cyan', 'beige', 'skyblue', 'lavender', 'gold', 'fuchsia', 'purple']
    i=1
    fig = plt.figure(1)
    # change the fontsize of the xtick and ytick labels and axes
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)
    for cl in n_cluster:
        if cl == -1:
            i = 0
        if len(n_cluster) <= 20:
            curr_c = color_order[i]
        else:
            curr_c = cm.tab20(float(cl) / len(n_cluster))
        id = clusters[clusters == cl].index
        curr_trajs = traj_2d[id, :]
        plt.scatter(curr_trajs[:, 0], curr_trajs[:, 1], label=cl, color=curr_c)
        plt.legend(scatterpoints=1, loc='best', shadow=False)
        i=i+1
    plt.xlabel(f'PC1 ({round(pc1,2)}%)')
    plt.ylabel(f'PC2 ({round(pc2,2)}%)')
    plt.savefig(f'{folder}/projection_PCA.png', bbox_inches='tight')
    plt.close()
    # plot_traj_pca(x, folder)

