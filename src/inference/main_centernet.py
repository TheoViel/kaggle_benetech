import json
import torch
import numpy as np
import pandas as pd

from data.dataset import CenterNetDataset
from data.transforms import get_transfos_centernet
from model_zoo.centernet import CenterNet
from util.torch import load_model_weights
from util.metrics import accuracy
from inference.predict import predict_centernet


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def kfold_inference(
    df,
    exp_folder,
    debug=False,
    use_tta=False,
    use_fp16=False,
):
    """
    Perform k-fold cross-validation for model inference on the validation set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        save (bool, optional): Whether to save the predictions. Defaults to True.
        use_tta (bool, optional): Whether to use test time augmentation. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.
        train (bool, optional): Whether to perform inference on the training set. Defaults to False.
        use_mt (bool, optional): Whether to use model teacher. Defaults to False.
        distilled (bool, optional): Whether to use distilled model. Defaults to False.
        n_soup (int, optional): Number of models to use for model soup. Defaults to 0.

    Returns:
        np.ndarray: Array containing the predicted probabilities for each class.
    """
    config = Config(json.load(open(exp_folder + "config.json", "r")))

    model = CenterNet(
        config.name,
        num_classes=config.num_classes,
        n_channels=config.n_channels,
    ).cuda()

    model = model.cuda().eval()

    preds = []
    for fold in config.selected_folds:
        print(f"\n- Fold {fold + 1}")

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)

        dataset = CenterNetDataset(
            df,
            transforms=get_transfos_centernet(augment=False, resize=config.resize),
        )

        pred = predict_centernet(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"],
            use_fp16=use_fp16,
        )

#         if "target" in df.columns:
#             acc = accuracy(df["target"].values, pred)
#             print(f"\n -> Accuracy : {acc:.4f}")
        
        preds.append(pred)

    return np.mean(preds, 0)
