NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = [""]
NUM_CLASSES = 250

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/Isolated-Sign-Language-Recognition"


GRAPH = [
    [42, 53, 52, 56, 59, 54, 57, 55, 58, 47],  # left head
    [42, 44, 43, 48, 51, 45, 49, 46, 50, 47],  # right head
    [63, 62, 61, 71, 72, 73, 79, 76, 66, 69, 63],  # outter lips
    [64, 70, 65, 60, 75, 80, 74, 78, 77, 67, 68, 64],  # inner lips
    # RIGHT HAND
    [41, 40, 39, 38],
    [37, 36, 35, 34],
    [33, 32, 31, 30],
    [29, 28, 27, 26],
    [25, 24, 23, 22, 21],
    [38, 34, 30, 26, 21, 38],
    # LEFT HAND
    [20, 19, 18, 17],
    [16, 15, 14, 13],
    [12, 11, 10, 9],
    [8, 7, 6, 5],
    [4, 3, 2, 1, 0],
    [17, 13, 9, 5, 0, 17],
    # Arms
    [82, 81, 83, 85, 87, 89, 91, 85],
    [82, 84, 86, 88, 90, 92, 86],
]


ANOMALIES = [
    # DUPLICATED STUFF
    'ae686738e744', 'c76f6d0d5239', '760c3fa4e3d9', 'c0c1f4046222', '3e568d136b85', '913447978a74', '2ff071a45cce', 'a9a07d74ee31',
    # MISSING or MISLABELED TICKS ANNOTS
    "36079df3b5b2", "3968efe9cbfc", "6ce4bc728dd5", "733b9b19e09a", "aa9df520a5f2", "d0cf883b1e13",
    # WEIRD
    "9f6b7c57e6cd", "e1034ff92655", "e796b10718bd", "f8bdbaf0b97d", "3ef41bbc82c3", "73cfbba65962", "872d1be39bae", "3ef41bbc82c3"
]
