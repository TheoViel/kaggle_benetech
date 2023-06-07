import gc
import re
import glob
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train_centernet import fit
from model_zoo.centernet import CenterNet

from data.preparation import prepare_centernet_data
from data.dataset import CenterNetDataset
from data.transforms import get_transfos_centernet

from util.torch import seed_everything, count_parameters, save_model_weights
from util.metrics import accuracy
from util.centernet import process_and_score


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
    train_dataset = CenterNetDataset(
        df_train,
        transforms=get_transfos_centernet(resize=config.resize, strength=config.aug_strength),

    )

    val_dataset = CenterNetDataset(
        df_val,
        transforms=get_transfos_centernet(augment=False, resize=config.resize),
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

    model = CenterNet(
        config.name,
        num_classes=config.num_classes,
        n_channels=config.n_channels,
        pretrained_weights=pretrained_weights,
        verbose=(config.local_rank == 0),
    ).cuda()

    if config.distributed:
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


def k_fold(config, log_folder=None, run=None):
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
    df_train, df_val = prepare_centernet_data(
        config.data_path, config.img_folder, use_extra=config.use_extra
    )

    if config.local_rank == 0:
        print(
            f"\n-------------   Train / Val Split  -------------\n"
        )

    seed_everything(int(re.sub(r"\W", "", config.name), base=36) % 2**31)

    pred_val = train(
        config, df_train, df_val, 0, log_folder=log_folder, run=run
    )

    if config.local_rank == 0:
        f1s = process_and_score(pred_val, df_val)
        print(f'-> Avg F1: {np.mean(f1s):.3f}  \t Avg F1==1: {np.mean(np.array(f1s) == 1):.3f}')

        if log_folder is not None:
            np.save(log_folder + "pred_val", pred_val)
            df_val.to_csv(log_folder + "df_val.csv", index=False)

            if run is not None:
                run["global/logs"].upload(log_folder + "logs.txt")
                run["global/pred_val"].upload(log_folder + "pred_val.npy")
                run["global/cv"] = np.mean(np.array(f1s) == 1)

    if config.fullfit:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            df_train = df.copy()

            train(
                config,
                df_train,
                df.tail(100).reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()

    return pred_val
