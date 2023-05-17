def post_process_preds(preds):
    try:
        graph = preds[0][0]
    except Exception:
        return preds

    # Points are inside the graph
    points = preds[3]
    margin = 10
    points = points[points[:, 0] > graph[0] - margin]
    points = points[points[:, 1] > graph[1] - margin]
    points = points[points[:, 2] < graph[2] + margin]
    points = points[points[:, 3] < graph[3] + margin]

    # Texts are below or left of the graph
    texts = preds[1]
    margin = 30
    texts = texts[
        (texts[:, 1] > graph[3] - margin)
        | (texts[:, 0] < graph[0] + margin)  # left  # bottom
    ]
    #     texts = texts[
    #         ((texts[:, 2] < graph[0]) & (texts[:, 3] > graph[1]) & (texts[:, 1] < graph[3])) |  # left
    #         ((texts[:, 1] > graph[3]) & (texts[:, 2] > graph[0]) & (texts[:, 0] < graph[2]))    # bottom
    #     ]

    # Ticks are on the axis
    ticks = preds[2]
    #     margin = 10
    #     ticks = ticks[
    #         ((np.abs((ticks[:, 2] + ticks[:, 2]) / 2 - graph[0]) < margin) & (ticks[:, 3] > graph[1]) & (ticks[:, 1] < graph[3])) |  # left
    #         ((np.abs((ticks[:, 1] + ticks[:, 3]) / 2 - graph[3]) < margin) & (ticks[:, 2] > graph[0]) & (ticks[:, 0] < graph[2]))    # bottom
    #     ]

    return [preds[0], texts, ticks, points]
