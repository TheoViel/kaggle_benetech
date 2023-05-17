import numpy as np


def my_assignment(mat):
    row_ind, col_ind = [], []
    for i in range(np.min(mat.shape)):
        row, col = np.unravel_index(np.argmin(mat), mat.shape)
        mat[row] = np.inf
        mat[:, col] = np.inf
        row_ind.append(row)
        col_ind.append(col)

    return row_ind, col_ind


def assign(ticks, labels, tol=2, mode="x"):
    if mode == "x":
        labels_x, labels_y = (labels[:, 0] + labels[:, 2]) / 2, labels[:, 1]
    else:
        labels_x, labels_y = labels[:, 2], (labels[:, 1] + labels[:, 3]) / 2

    labels_xy = np.stack([labels_x, labels_y], -1)
    #     print(labels_xy.shape)

    ticks_x, ticks_y = (ticks[:, 0] + ticks[:, 2]) / 2, (ticks[:, 1] + ticks[:, 3]) / 2
    ticks_xy = np.stack([ticks_x, ticks_y], -1)

    #     print(ticks_xy.shape)

    cost_matrix = np.sqrt(((ticks_xy[:, None] - labels_xy[None]) ** 2).sum(-1))

    #     print(np.min(cost_matrix))
    if mode == "x":  # penalize y_label < y_tick
        cost_matrix += (
            ((ticks_y[:, None] - labels_y[None]) > 0) * np.min(cost_matrix) * tol
        )
    else:  # penalize x_tick < x_label
        cost_matrix += (
            ((ticks_x[:, None] - labels_x[None]) < 0) * np.min(cost_matrix) * tol
        )

    row_ind, col_ind = my_assignment(cost_matrix.copy())

    #     print(row_ind, col_ind)

    ticks_assigned, labels_assigned = [], []

    for tick_idx, label_idx in zip(row_ind, col_ind):
        #         print(cost_matrix[tick_idx, label_idx])
        if cost_matrix[tick_idx, label_idx] < max(tol * 5, tol * np.min(cost_matrix)):
            ticks_assigned.append(ticks[tick_idx])
            labels_assigned.append(labels[label_idx])

    return np.array(ticks_assigned), np.array(labels_assigned)


def restrict_on_line(preds, margin=5, cat=False):
    try:
        graph = preds[0][0]
        x_axis, y_axis = graph[0], graph[3]
    except Exception:
        x_axis, y_axis = 0, 0
   

    ticks = preds[2]
    ticks_x, ticks_y = (ticks[:, 0] + ticks[:, 2]) / 2, (ticks[:, 1] + ticks[:, 3]) / 2

    #     print(x_axis, y_axis)
    #     print(ticks_x)
    #     print(ticks_y)

    dists_x = ticks_x - x_axis
    dists_y = ticks_y - y_axis

    best_x = dists_x[np.argmax([(np.abs(dists_x - d) < margin).sum() for d in dists_x])]
    best_y = dists_y[np.argmax([(np.abs(dists_y - d) < margin).sum() for d in dists_y])]

    #     print(dists_x - best_x)
    #     print(dists_y - best_y)
    y_ticks = ticks[np.abs(dists_x - best_x) < margin]  # similar x
    x_ticks = ticks[np.abs(dists_y - best_y) < margin]  # similar y

#     print(x_ticks)
#     print(y_ticks)

    # Pair with labels
    labels = preds[1]

    x_ticks, x_labels = assign(x_ticks.copy(), labels.copy())
    y_ticks, y_labels = assign(y_ticks.copy(), labels.copy(), mode="y")

    # Reorder
    order_x = np.argsort(x_ticks[:, 0])
    x_ticks = x_ticks[order_x]
    x_labels = x_labels[order_x]

    order_y = np.argsort(y_ticks[:, 1])[::-1]
    y_ticks = y_ticks[order_y]
    y_labels = y_labels[order_y]

    if not cat:
        return [preds[0], x_labels, y_labels, x_ticks, y_ticks, preds[3]]

    labels = np.unique(np.concatenate([x_labels, y_labels]), axis=0)
    ticks = np.unique(np.concatenate([x_ticks, y_ticks]), axis=0)

    return [preds[0], labels, ticks, preds[3]]