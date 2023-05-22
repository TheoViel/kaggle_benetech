import glob
import pandas as pd
from params import CLASSES, ANOMALIES


def prepare_data(data_path="../input/", img_folder="train/images/"):
    df = pd.read_csv(data_path + 'df_train.csv')
    df = df[~df['id'].isin(ANOMALIES)].reset_index(drop=True)
    
    df_split = pd.read_csv(data_path + 'df_split.csv')
    df = df.merge(df_split)
    
    df['path'] = data_path + img_folder + df['id'] + ".jpg"
    
    df = df[['source', "chart-type", "split", "path"]].copy()
    
    df['target'] = df['chart-type'].apply(lambda x: CLASSES.index(x))
    
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
