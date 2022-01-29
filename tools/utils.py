import os
import json
import yaml
import argparse

from typing import Dict, Tuple


def open_conf(conf_path: str) -> dict:
    """Loads the config JSON.
    Args:
        conf_path (str): config file path.
    Returns:
        dict: config values as dict.
    """
    with open(os.path.join(os.getcwd(), conf_path), "r") as f:
        conf = json.load(f)

    return conf


def default_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.model = "bert"
    args.dataset = "tweet_eval"
    args.tokenizer = "bert-base-cased"
    args.metrics = "acc"
    args.loggers = "Board"
    args.learning_rate = 1e-3
    args.batch_size = 8
    args.workers = 4

    args.auto_lr_find = False

    return args


def get_checkpoint_hparams(
    path: str, checkpoint_idx: int = -1
) -> Tuple[str, str, Dict]:
    """Read a YAML file from Pytorch Lightning to get the info of
    the checkpoint desired.
    Args:
        path (str): to the checkpoint
        checkpoint_idx (int, optional): In case of having several
        checkpoints. Defaults to -1.
    Returns:
        Tuple[str, str, Dict]: the model name, the checkpoint and
        the hyperparams.
    """
    path = path[:-1] if path[-1] == "/" else path
    all_checks = os.listdir(f"{path}/checkpoints")
    checkpoint = f"{path}/checkpoints/{all_checks[checkpoint_idx]}"
    model = path.split("/")[-2]

    with open(f"{path}/hparams.yaml", "r") as y_file:
        hparams = yaml.safe_load(y_file)

    return model, checkpoint, hparams
