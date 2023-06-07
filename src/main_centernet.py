import os
import time
import torch
import warnings
import argparse
import pandas as pd

from params import DATA_PATH
from util.torch import init_distributed
from util.logger import create_logger, save_config, prepare_log_folder, init_neptune
from util.centernet import process_and_score


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold number",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Images
    img_folder = "v13/"
    data_path = DATA_PATH
    aug_strength = 1
    resize = (512, 512)
    use_extra = True
    dots_only = True

    # k-fold
    k = 4
    folds_file = None
    selected_folds = [0]

    # Model
    name = "resnet18"  # "eca_nfnet_l2"  # "tf_efficientnetv2_s" "eca_nfnet_l1"
    pretrained_weights = None
    num_classes = 3
    n_channels = 3
    drop_rate = 0.
    drop_path_rate = 0.
    syncbn = False

    # Training
    loss_config = {
        "name": "centerloss",  # bce ?
        "smoothing": 0.0,
        "activation": "sigmoid",
        "aux_loss_weight": 0.,
    }

    data_config = {
        "batch_size": 8,
        "val_bs": 32,
        "mix": "cutmix",
        "mix_proba": 0,
        "mix_alpha": 4.0,
        "num_classes": num_classes,
        "additive_mix": False,
    }

    optimizer_config = {
        "name": "Ranger",
        "lr": 5e-3,
        "warmup_prop": 0.1,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.0,
        "weight_decay": 0,  # 1e-2,
    }

    epochs = 20
    use_fp16 = True

    verbose = 1
    verbose_eval = 200

    fullfit = False
    n_fullfit = 1



if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = args.fold if args.fold > -1 else args.device
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    if not log_folder:
        from params import LOG_PATH

        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH)

    if args.model:
        config.name = args.model

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    try:
        print(torch_performance_linter)  # noqa
        if config.local_rank == 0:
            print("Using TPL\n")
        run = None
        config.epochs = 1
        log_folder = None
        df = df.head(10000)
    except Exception:
        run = None
        if config.local_rank == 0:
            run = init_neptune(Config, log_folder)

            if args.fold > -1:
                config.selected_folds = [args.fold]
                create_logger(directory=log_folder, name=f"logs_{args.fold}.txt")
            else:
                create_logger(directory=log_folder, name="logs.txt")

            save_config(config, log_folder + "config.json")

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    from training.main_centernet import k_fold
    k_fold(Config, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
