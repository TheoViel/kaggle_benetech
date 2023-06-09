import numpy as np


def post_process_arrow(preds, max_dist=4, verbose=0):
    if not len(preds[0]):
        return preds

    chart = preds[0][0]
        
    preds_pp = []
    for point in preds[-1]:
        xc = (point[0] + point[2]) / 2
        yc = (point[1] + point[3]) / 2

        if np.sqrt((xc - chart[0]) ** 2 + (yc - chart[1]) ** 2) < max_dist:
            if verbose:
                print('y axis arrow !')
            continue
        if np.sqrt((xc - chart[1]) ** 2 + (yc - chart[3]) ** 2) < max_dist:
            if verbose:
                print('x axis arrow !')
            continue
            
        preds_pp.append(point)
        
    preds[-1] = np.array(preds_pp)
    return preds


def batched_iou(A, B):
    A = A.copy()
    B = B.copy()
    
    A = A[None].repeat(B.shape[0], 0)
    B = B[:, None].repeat(A.shape[1], 1)
        
    low = np.s_[...,:2]
    high = np.s_[...,2:]

    A, B = A.copy(), B.copy()
    A[high] += 1
    B[high] += 1

    intrs = (np.maximum(0, np.minimum(A[high], B[high]) - np.maximum(A[low], B[low]))).prod(-1)
    ious =  intrs / ((A[high] - A[low]).prod(-1) + (B[high] - B[low]).prod(-1) - intrs)
    
    return ious


def post_process_point_as_tick(preds, marker_conf=None, th=0.5, max_dist=4, max_dist_o=0, verbose=0):
    ticks = np.concatenate([preds[2], preds[1]])
    points = preds[3]
    
    if marker_conf is None:
        marker_conf = np.zeros(len(points))
    
    try:
        chart = preds[0][0]
        x_o, y_o = chart[0], chart[3]
#         print(x_o, y_o)
    except:
        x_o, y_o = (0, 0)
    
    ious = batched_iou(ticks, points)
    
    ticks_x = (ticks[:, 0] + ticks[:, 2]) / 2
    ticks_y = (ticks[:, 1] + ticks[:, 3]) / 2
    
    points_x = (points[:, 0] + points[:, 2]) / 2
    points_y = (points[:, 1] + points[:, 3]) / 2
    
    dists = np.sqrt(
        (ticks_x[None] - points_x[:, None]) ** 2 + 
        (ticks_y[None] - points_y[:, None]) ** 2
    )
#     print(dists.shape, ious.shape)
#     print(dists.min())

    if verbose:
#         print("IoU", np.max(ious, 1))
#         print((np.abs(points_x - x_o) + np.abs(points_y - y_o)))
        if len(np.argwhere(np.max(ious, 1) > th)):
            print('IOU : Removing', np.argwhere(np.max(ious, 1) > th).flatten())
            print('Confs :', marker_conf[np.argwhere(np.max(ious, 1) > th).flatten()])
        if len(np.argwhere(np.min(dists, 1) < max_dist)):
            print('Dist : Removing', np.argwhere(np.min(dists, 1) < max_dist).flatten())
            print('Confs :', marker_conf[np.argwhere(np.min(dists, 1) < max_dist).flatten()])
    
#     print(ious.shape, points.shape)
    points = points[
        ((np.max(ious, 1) < th) | (np.min(dists, 1) > max_dist)) |
        ((np.abs(points_x - x_o) + np.abs(points_y - y_o)) < max_dist_o) |
        (marker_conf > 0.9)
    ]

    preds[3] = points
    return preds
