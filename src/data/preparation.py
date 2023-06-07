import glob
import pandas as pd
from pathlib import Path

from params import CLASSES, ANOMALIES


def limit_training_samples(df, lims={}, key='chart-type'):
    df_train = df[df['split'] == "train"]
    df_val = df[df['split'] != "train"]

    limited_dfs = [df_val]
    
    for c, dfg in df_train.groupby(key):
        lim = lims.get(c, None)
        if lim is None:
            lim = lims.get("*", None)
        if lim is None:
            lim = 1000000

        limited_dfs.append(dfg.head(lim))
        
    return pd.concat(limited_dfs, ignore_index=True)


def prepare_dots(data_path="../input/", oversample=100):
    dfs = []
        
    df = pd.DataFrame({"path": glob.glob(data_path + 'dots/*')})
    df['id'] = df['path'].apply(lambda x: Path(x).stem)
    df['source'] = "extracted"
    df['chart-type'] = "dot"
    df['split'] = "val"
    df['target'] = 1
    dfs.append(df)
   
    df = pd.DataFrame({"path": glob.glob(data_path + 'dots_2/*')})
    df['id'] = df['path'].apply(lambda x: Path(x).stem)
    df['source'] = "extracted"
    df['chart-type'] = "dot"
    df['split'] = "train"
    df['target'] = 1
    
    for _ in range(oversample):
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)


def prepare_data(data_path="../input/", img_folder="train/images/"):
    df = pd.read_csv(data_path + 'df_train.csv')
    df = df[~df['id'].isin(ANOMALIES)].reset_index(drop=True)
    
    df_split = pd.read_csv(data_path + 'df_split.csv')
    df = df.merge(df_split)
    
    df['path'] = data_path + img_folder + df['id'] + ".jpg"
    df = df[['source', "chart-type", "split", "path"]].copy()
    df['target'] = df['chart-type'].apply(lambda x: CLASSES.index(x))
    
    df.loc[df["source"] == "extracted", "split"] = "val"
    df.loc[df["source"] != "extracted", "split"] = "train"
    return df


def prepare_gen_data(data_path="../input/", img_folder="generated/"):
    gen = glob.glob(data_path + img_folder + "*/*.jpg")
    df_gen = pd.DataFrame({"path": gen})
    
    df_gen['chart-type'] = df_gen['path'].apply(lambda x: x.split('/')[-2])
    df_gen['target'] = df_gen['chart-type'].apply(lambda x: CLASSES.index(x))
    df_gen['split'] = "train"
    df_gen['source'] = "generated"

    df_gen = df_gen[['source', 'chart-type', 'split', 'path', 'target']]
    return df_gen


def prepare_xqa_data(data_path="../input/", img_folder="xQA/curated/"):
    gen = glob.glob(data_path + img_folder + "*/*.jpg")
    df_gen = pd.DataFrame({"path": gen})

    df_gen['chart-type'] = df_gen['path'].apply(lambda x: x.split('/')[-2])
    df_gen['target'] = df_gen['chart-type'].apply(lambda x: CLASSES.index(x))
    df_gen['split'] = "train"
    df_gen['source'] = "xqa"

    df_gen = df_gen[['source', 'chart-type', 'split', 'path', 'target']]
    
    return df_gen


def prepare_centernet_data(data_path="../input/", img_folder="v13_sim/", use_extra=True):
    df_val = pd.DataFrame({"path": glob.glob(f'{data_path}/{img_folder}/val2017/*')})
    df_val['id'] = df_val['path'].apply(lambda x: Path(x).stem)
    df_val['source'] = "extracted"
    df_val['gt_path'] = f'{data_path}/{img_folder}/labels/valid/' + df_val['id'] + '.txt'
    
    df = pd.DataFrame({"path": glob.glob(f'{data_path}/{img_folder}/val2017/*')})

    df = pd.DataFrame({"path": glob.glob(f'{data_path}/{img_folder}/train2017/*')})
    df['id'] = df['path'].apply(lambda x: Path(x).stem)
    df['source'] = "extracted"
    df['gt_path'] = f'{data_path}/{img_folder}/labels/train/' + df['id'] + '.txt'
    
    if not use_extra:
        not_extra = pd.read_csv(data_path + 'df_train.csv')['id'].values
        df = df[df['id'].isin(not_extra)].reset_index(drop=True)
        
#     dft = pd.read_csv(data_path + 'df_train.csv')
#     dots = dft[dft['chart-type'] == "dot"].id.values
#     df = df[df['id'].isin(dots)].reset_index(drop=True)

    return df, df_val
