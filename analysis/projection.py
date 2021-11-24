from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pca_coeffs(x):
    """
    It applies PCA projection using 2 components.
    :param x: the dataset
    :return: the projection of the dataset in 2d
    """
    pca_model = PCA(n_components=2)
    pca_model.fit(x)
    traj_2d = pca_model.fit_transform(x)
    return traj_2d


def plot_traj_pca(x, folder):
    """
    It generates the plot with PCA projection for 2 components.
    :param x: the dataset
    :param folder: the folder to save the image
    """
    # PCA
    traj_2d = pca_coeffs(x)
    fig = plt.figure(1)
    plt.scatter(traj_2d[:, 0], traj_2d[:, 1])
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.savefig(f'{folder}_projection_PCA.png', bbox_inches='tight')
    plt.close()


def plot_coeffs_traj(x, clusters, folder):
    """
    It generates the plot with PCA projection for 2 components colored based on the clustering results.
    :param x: the dataset
    :param clusters: the dataset labels
    :param folder: the folder to save the image
    """
    traj_2d = pca_coeffs(x)
    n_cluster = clusters.unique()

    fig = plt.figure(1)
    for cl in n_cluster:
        id = clusters[clusters == cl].index
        curr_trajs = traj_2d[id, :]
        plt.scatter(curr_trajs[:, 0], curr_trajs[:, 1], label=cl)
        plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.savefig(f'{folder}/projection_PCA.png', bbox_inches='tight')
    plt.close()

