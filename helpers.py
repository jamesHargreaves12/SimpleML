import numpy as np


def getPartition(partitionNumber, totalNumberOfPartitions, X_train_real, y_train_real):
    N = X_train_real.shape[0]
    start_index = N * partitionNumber // totalNumberOfPartitions
    end_index = N * (partitionNumber + 1) // totalNumberOfPartitions
    X_before = X_train_real[:start_index] if start_index > 0 else np.empty((0, 28, 28))
    X_after = X_train_real[end_index:] if end_index < N else np.empty((0, 28, 28))
    y_before = y_train_real[:start_index] if start_index > 0 else np.empty((0))
    y_after = y_train_real[end_index:] if end_index < N else np.empty((0))
    X_train = np.concatenate((X_before, X_after))
    y_train = np.concatenate((y_before, y_after))
    return X_train, y_train
