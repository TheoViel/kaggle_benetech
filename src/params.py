NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["vertical_bar", "dot", "line", "scatter", "horizontal_bar"]
NUM_CLASSES = len(CLASSES)

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/Benetech"

ANOMALIES = [
    # DUPLICATED STUFF
    "ae686738e744",
    "c76f6d0d5239",
    "760c3fa4e3d9",
    "c0c1f4046222",
    "3e568d136b85",
    "913447978a74",
    "2ff071a45cce",
    "a9a07d74ee31",
    # MISSING or MISLABELED TICKS ANNOTS
    "36079df3b5b2",
    "3968efe9cbfc",
    "6ce4bc728dd5",
    "733b9b19e09a",
    "aa9df520a5f2",
    "d0cf883b1e13",
    # WEIRD
    "9f6b7c57e6cd",
    "e1034ff92655",
    "e796b10718bd",
    "f8bdbaf0b97d",
    "3ef41bbc82c3",
    "73cfbba65962",
    "872d1be39bae",
    "3ef41bbc82c3",
]
