# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def predict_OneVsOne(confidence, classes):
    # for SVC, NuSVC
    n_samples, n_w = confidence.shape
    votes = np.zeros((n_samples, n_w))
    k = 0
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes[(i + 1):]):
            compared_classes = np.array([class1, class2])
            comparison = confidence[:, k] < 0
            votes[:, k] = compared_classes[comparison.astype(int)]
            k += 1
    summed_votes = np.array([np.sum(votes == c, axis=1) for c in classes]).T
    y_pred = predict_OneVsRest(summed_votes, classes)
    return y_pred


def predict_OneVsRest(confidence, classes):
    # for LinearSVC
    return np.array(classes[confidence.argmax(axis=1)])
