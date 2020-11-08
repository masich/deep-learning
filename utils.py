import numpy as np


def normalize(data):
    data = (data - np.min(data))
    return data / np.max(data)


def pca(data, dim=0):
    """
    Principal Component Analysis.
    """
    # Mean value from the dataset
    mean = np.mean(data, axis=0)
    data -= mean

    # Covariance matrix
    cov = np.cov(np.transpose(data))

    # Compute eigenvalues and sort in descending order
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sort_indices = np.argsort(eigenvalues)
    sort_indices = sort_indices[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    eigenvalues = eigenvalues[sort_indices]

    if dim > 0:
        eigenvectors = eigenvectors[:, :dim]

    # Construct a new dataset
    x = np.dot(np.transpose(eigenvectors), np.transpose(data))
    # Simulate an old dataset based on the calculated vectors
    y = np.transpose(eigenvectors @ x) + mean
    return x, y, eigenvalues, eigenvectors
