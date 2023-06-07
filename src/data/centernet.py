import numpy as np

# input_size = 1024
# IN_SCALE = 1024 // input_size 
# MODEL_SCALE = 4


# Make heatmaps using the utility functions from the centernet repo
def draw_msra_gaussian(heatmap, center, sigma=2):
    tmp_size = sigma * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_regmap = regmap[:, y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    masked_reg = reg[:, radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top : y + bottom, x - left : x + right] = masked_regmap
    return regmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# Wrapped heatmap function
def make_hm_regr(target, shape, model_scale=1, in_scale=1):
    # make output heatmap for single class
    w, h = shape[:2]
    hm = np.zeros((w, h))
    # make regr heatmap
    regr = np.zeros([2, h, w])

    if len(target) == 0:
        return hm, regr

#     try:
#         center = np.array(
#             [
#                 target["x"] + target["w"] // 2,
#                 target["y"] + target["h"] // 2,
#                 target["w"],
#                 target["h"],
#             ]
#         ).T
#     except:
#         center = np.array(
#             [
#                 int(target["x"] + target["w"] // 2),
#                 int(target["y"] + target["h"] // 2),
#                 int(target["w"]),
#                 int(target["h"]),
#             ]
#         ).T.reshape(1, 4)
        
    center = target.copy()
    center[:, [0, 2]] *= h
    center[:, [1, 3]] *= w

    # make a center point
    # try gaussian points.
    for c in center:
        hm = draw_msra_gaussian(
            hm,
            [
                int(c[0]) // model_scale // in_scale,
                int(c[1]) // model_scale // in_scale,
            ],
            sigma=np.clip(c[2] * c[3] // 2000, 2, 4),
        )

    # convert targets to its center.
    regrs = target[:, 2:] / in_scale
    
    # plot regr values to mask
    for r, c in zip(regrs, center):
        for i in range(-2, 3):
            for j in range(-2, 3):
                try:
                    regr[
                        :,
                        int(c[0]) // model_scale // in_scale + i,
                        int(c[1]) // model_scale // in_scale + j,
                    ] = r
                except:
                    pass
    regr = regr.transpose(0, 2, 1)
    return hm, regr
