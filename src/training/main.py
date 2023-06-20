import gc
import re
import glob
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models import define_model

from data.dataset import ClsDataset
from data.transforms import get_transfos

from util.torch import seed_everything, count_parameters, save_model_weights
from util.metrics import accuracy


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (neptune.Run): Nepture run. Defaults to None.

    Returns:
        np array [len(df_val) x num_classes]: Validation predictions.
    """
    train_dataset = ClsDataset(
        df_train,
        transforms=get_transfos(resize=config.resize, strength=config.aug_strength),

    )

    val_dataset = ClsDataset(
        df_val,
        transforms=get_transfos(augment=False, resize=config.resize),
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[
                0
            ]
    else:
        pretrained_weights = None

    model = define_model(
        config.name,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        pretrained_weights=pretrained_weights,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        use_gem=config.use_gem,
        reduce_stride=config.reduce_stride,
        verbose=(config.local_rank == 0),
    ).cuda()

    if config.distributed:
        if config.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=config.syncbn,
        )

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.name.split('/')[-1]}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return pred_val


def k_fold(config, df, df_extra=None, log_folder=None, run=None):
    """
    Trains a k-fold.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (pandas dataframe or None, optional): Extra metadata. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (None or Nepture run): Nepture run. Defaults to None.

    Returns:
        np array [len(df) x num_classes]: Oof predictions.
    """

    df_train = df[df['split'] == "train"].reset_index(drop=True)
    df_val = df[df['split'] != "train"].reset_index(drop=True)
    df_val = df_val.drop_duplicates(keep="first", subset="path").reset_index(drop=True)

    if config.local_rank == 0:
        print("\n-------------   Train / Val Split  -------------\n")

    seed_everything(int(re.sub(r"\W", "", config.name), base=36) % 2**31)

    pred_val = train(
        config, df_train, df_val, 0, log_folder=log_folder, run=run
    )

    if config.local_rank == 0:
        acc = accuracy(df_val["target"].values, pred_val)
        print(f"\n\n -> CV Accuracy : {acc:.4f}")

        if log_folder is not None:
            np.save(log_folder + "pred_val", pred_val)
            df_val.to_csv(log_folder + "df_val.csv", index=False)

            if run is not None:
                run["global/logs"].upload(log_folder + "logs.txt")
                run["global/pred_val"].upload(log_folder + "pred_val.npy")
                run["global/cv"] = acc

    if config.fullfit:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            df_train_ff = pd.concat(
                [df_train] + [df_val] * config.oversample_extracted,
                ignore_index=True
            )

            train(
                config,
                df_train_ff,
                df_val,
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()

    return pred_val
