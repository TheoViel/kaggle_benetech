import glob
import pandas as pd
from pathlib import Path

from params import CLASSES, ANOMALIES


def limit_training_samples(df, lims={}, key="chart-type"):
    """
    Limit the number of training samples based on given limits for specific categories.

    Args:
        df (pandas DataFrame): Original DataFrame.
        lims (dict, optional): Limits for specific categories. Defaults to {}.
        key (str, optional): Key column used for grouping and applying limits. Defaults to "chart-type".

    Returns:
        pandas DataFrame: Limited DataFrame.
    """
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] != "train"]

    limited_dfs = [df_val]

    for c, dfg in df_train.groupby(key):
        lim = lims.get(c, None)
        if lim is None:
            lim = lims.get("*", None)
        if lim is None:
            lim = 1000000

        limited_dfs.append(dfg.head(lim))

    return pd.concat(limited_dfs, ignore_index=True)


def prepare_dots(data_path="../input/", oversample=100, oversample_val=1):
    """
    Prepare the dots dataframe by creating a DataFrame with paths, labels, and other information.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        oversample (int, optional): Number of times to oversample the training samples. Defaults to 100.
        oversample_val (int, optional): Number of times to oversample the validation samples. Defaults to 1.

    Returns:
        pandas DataFrame: Prepared dots dataframe.
    """
    dfs = []

    df = pd.DataFrame({"path": glob.glob(data_path + "dots/*")})
    df["id"] = df["path"].apply(lambda x: Path(x).stem)
    df["source"] = "extracted"
    df["chart-type"] = "dot"
    df["split"] = "val"
    df["target"] = 1

    for _ in range(oversample_val):
        dfs.append(df)

    df = pd.DataFrame({"path": glob.glob(data_path + "dots_2/*")})
    df["id"] = df["path"].apply(lambda x: Path(x).stem)
    df["source"] = "extracted"
    df["chart-type"] = "dot"
    df["split"] = "train"
    df["target"] = 1

    for _ in range(oversample):
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def prepare_data(data_path="../input/", img_folder="train/images/"):
    df = pd.read_csv(data_path + "df_train.csv")
    df = df[~df["id"].isin(ANOMALIES)].reset_index(drop=True)

    df_split = pd.read_csv(data_path + "df_split.csv")
    df = df.merge(df_split)

    df["path"] = data_path + img_folder + df["id"] + ".jpg"
    df = df[["source", "chart-type", "split", "path"]].copy()
    df["target"] = df["chart-type"].apply(lambda x: CLASSES.index(x))

    df.loc[df["source"] == "extracted", "split"] = "val"
    df.loc[df["source"] != "extracted", "split"] = "train"
    return df


def prepare_gen_data(data_path="../input/", img_folder="generated/"):
    gen = glob.glob(data_path + img_folder + "*/*.jpg")
    df_gen = pd.DataFrame({"path": gen})

    df_gen["chart-type"] = df_gen["path"].apply(lambda x: x.split("/")[-2])
    df_gen["target"] = df_gen["chart-type"].apply(lambda x: CLASSES.index(x))
    df_gen["split"] = "train"
    df_gen["source"] = "generated"

    df_gen = df_gen[["source", "chart-type", "split", "path", "target"]]
    return df_gen
