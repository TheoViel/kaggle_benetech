import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter


def cluster_on_x(dots, w, plot=False):
    xs = (dots[:, 0] + dots[:, 2]) / 2
    ys = (dots[:, 1] + dots[:, 3]) / 2

    dbscan = DBSCAN(min_samples=1, eps=0.01 * w)
    dbscan.fit(xs[:, None])
    labels = dbscan.labels_

    centers = []
#     plot = True
    for l in np.unique(labels):
        centers.append(xs[labels == l].mean())

        if plot:
            plt.scatter(
                xs[labels == l],
                -ys[labels == l],
                label=f"Cluster {l}",
            )
    if plot:
        plt.legend()
        plt.show()
        
    labels = np.array(labels)
    centers = np.array(centers)
#     print(labels)
#     print(centers)
    clusters_y = [np.sort(ys[labels == i])[::-1] for i in np.unique(labels)]

    first = np.median([c[0] for c in clusters_y])
    second = np.median([c[1] for c in clusters_y if len(c) > 1])
    third = np.median([c[2] for c in clusters_y if len(c) > 2])
    
    if len([c[1] for c in clusters_y if len(c) > 2]) > 2:
        delta = (first - third) / 2
    else:
        delta = (first - second)
    
#     print(delta)
#     print([first - c.min() for c in clusters_y])
    
    counts = [np.round((first - c.min()) / (delta) + 1) for c in clusters_y]
    
    clusters = dict(zip(np.unique(labels), counts))
#     print(clusters)
#     print(Counter(labels))

    return centers, clusters  # Counter(labels)


def my_assignment(mat):
    row_ind, col_ind = [], []
    for i in range(np.min(mat.shape)):
        row, col = np.unravel_index(np.argmin(mat), mat.shape)
        value = mat[row, col]
        if value < 20:
            mat[row] = np.inf
            mat[:, col] = np.inf
            row_ind.append(row)
            col_ind.append(col)

    return row_ind, col_ind


def assign_dots(labels, centers, tol=10, retrieve_missing=False, verbose=0):
    labels_x = (labels[:, 0] + labels[:, 2]) / 2
    cost_matrix = np.abs(labels_x[:, None] - centers[None])

    row_ind, col_ind = my_assignment(cost_matrix.copy())

    mapping = dict(zip(row_ind, col_ind))

    if not retrieve_missing:
        return mapping, []

    # Unassigned labels
    # mapping.update({k: -1 for k in len(labels) if k not in mapping.keys()})

    # Unassigned dots
    unassigned = [k for k in range(len(centers)) if k not in mapping.values()]
    centers_unassigned = centers[unassigned]

    #     print(centers_unassigned)

    if not len(unassigned):
        return mapping, []

    yc = (
        ((labels[:, 1] + labels[:, 3]) / 2)
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    w = (
        (labels[:, 2] - labels[:, 0])
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    h = (
        (labels[:, 3] - labels[:, 1])
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    xc = centers_unassigned[:, None]

    #     print(xc.shape, yc.shape)

    retrieved = np.concatenate(
        [xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], 1
    ).astype(int)

    #     print(retrieved)

    mapping.update({len(labels) + i: k for i, k in enumerate(unassigned)})

    return mapping, retrieved


def restrict_labels_x(preds, margin=5):
    try:
        graph = preds[0][0]
        x_axis, y_axis = graph[0], graph[3]
    except Exception:
        x_axis, y_axis = 0, 0

    labels = preds[1]

    if not len(labels):
        return [preds[0], np.empty((0, 4)), np.empty((0, 4)), preds[3]]

    labels_x, labels_y = (labels[:, 0] + labels[:, 2]) / 2, (
        labels[:, 1] + labels[:, 3]
    ) / 2

    dists_x = labels_x - x_axis
    dists_y = labels_y - y_axis

    best_x = dists_x[np.argmax([(np.abs(dists_x - d) < margin).sum() for d in dists_x])]
    best_y = dists_y[np.argmax([(np.abs(dists_y - d) < margin).sum() for d in dists_y])]

    y_labels = labels[np.abs(dists_x - best_x) < margin]  # similar x  # noqa
    x_labels = labels[np.abs(dists_y - best_y) < margin]  # similar y

    return [preds[0], x_labels, np.empty((0, 4)), preds[3]]


def constraint_size(dots, coef=0, margin=1):
    ws = dots[:, 2] - dots[:, 0]
    hs = dots[:, 3] - dots[:, 1]

    median_w = np.median(ws[:10])
    median_h = np.median(hs[:10])

    dots = dots[
        (np.abs(ws - median_w) < (median_w * coef + margin))
        & (np.abs(hs - median_h) < (median_h * coef + margin))
    ]
    return dots
