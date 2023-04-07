import json
import pandas as pd

from params import DATA_PATH


def prepare_folds(k):
    from sklearn.model_selection import StratifiedGroupKFold
    K = k

    df = pd.read_csv(DATA_PATH + "train.csv")

    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=42)
    splits = sgkf.split(df, y=df['sign'], groups=df['participant_id'])

    df['fold'] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["participant_id", "sequence_id", "fold"]]
    df_folds.to_csv(f"../input/folds_{K}.csv", index=False)


def prepare_data(data_path="../input/", processed_folder=""):
    df = pd.read_csv(data_path + "train.csv")
    classes = json.load(open(data_path + "sign_to_prediction_index_map.json", "r"))

    df["target"] = df["sign"].map(classes)
    df['path'] = data_path + df['path']
    
    df['processed_path'] = DATA_PATH + processed_folder + df['participant_id'].astype(str) + "_" + df['sequence_id'].astype(str) + ".npy"

    return df


def get_annotations(annotations_path="../input/train/annotations"):
    annotations = []
    for annotation_path in tqdm(os.listdir(annotations_path)):
        with open(f"{annotations_path}/{annotation_path}") as annotation_f:
            annots = json.load(annotation_f)
            annots['id'] = annotation_path[:-5]
            annotations.append(annots)
    return annotations
