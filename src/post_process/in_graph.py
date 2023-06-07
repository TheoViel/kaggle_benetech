import numpy as np


def post_process_preds(preds, margin_pt=10, margin_text=30):
    try:
        graph = preds[0][0]
    except Exception:
        return preds

    # Points are inside the graph
    points = preds[3]
    margin = margin_pt
    points = points[points[:, 0] > graph[0] - margin]
    points = points[points[:, 1] > graph[1] - margin]
    points = points[points[:, 2] < graph[2] + margin]
    points = points[points[:, 3] < graph[3] + margin]
    
#     # Points on the axes are ticks
#     xc = (points[:, 0] + points[:, 2]) / 2
#     yc = (points[:, 1] + points[:, 3]) / 2
#     points = points[(np.abs(xc - graph[0]) > 1) & (np.abs(yc - graph[3]) > 1)]

    # Texts are below or left of the graph
    texts = preds[1]
    margin = margin_text
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


def post_process_preds_dots(preds, margin_pt=10, margin_text=30):
    try:
        graph = preds[0][0]
    except Exception:
        return preds

    # Points are inside the graph
    points = preds[3]
    margin = margin_pt
#     print(graph)
    points = points[points[:, 2] > graph[0] + margin]
    points = points[points[:, 1] > graph[1] - margin]
    points = points[points[:, 2] < graph[2] + margin]
    points = points[points[:, 1] < graph[3] - margin]
    
    if not len(points):
        points = preds[3]
    
#     # Points on the axes are ticks
#     xc = (points[:, 0] + points[:, 2]) / 2
#     yc = (points[:, 1] + points[:, 3]) / 2
#     points = points[(np.abs(xc - graph[0]) > 1) & (np.abs(yc - graph[3]) > 1)]

    # Texts are below & right
    texts = preds[1]
    margin = margin_text
    
#     print(graph)
    
    texts = texts[
        ((texts[:, 2] + texts[:, 0]) / 2) > (graph[0] - margin)
    ]
    texts = texts[
        ((texts[:, 3] + texts[:, 1]) / 2) > (graph[3] - margin)
    ]
    if not len(texts):
        texts = preds[1]

    ticks = preds[2]


    return [preds[0], texts, ticks, points]
