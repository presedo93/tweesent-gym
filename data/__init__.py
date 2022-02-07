import pytorch_lightning as pl

from data.tweeteval import TweetEvalModule, TWEET_DESC
from typing import Any, Dict, KeysView


DATASETS: Dict[str, Dict[str, Any]] = {
    "tweet_eval": {"set": TweetEvalModule, "desc": TWEET_DESC}
}


def available_datasets() -> KeysView[str]:
    return DATASETS.keys()


def dataset_picker(model_name: str) -> pl.LightningDataModule:
    try:
        return DATASETS[model_name.lower()]["set"]
    except KeyError:
        return DATASETS["tweet_eval"]["set"]


def desc_dataset(data_name: str) -> str:
    try:
        return DATASETS[data_name.lower()]["desc"]
    except KeyError:
        return DATASETS["tweet_eval"]["desc"]
