import torch
import datasets
import pytorch_lightning as pl

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union


class TweetEvalModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer: str, *, workers: int = 4, batch_size: int = 16, **kwargs
    ):
        super().__init__()
        # Tokenizer from pre-trained model.
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        # Train/test stage parameters.
        self.batch_size = batch_size
        self.workers = workers

    def prepare_data(self) -> None:
        # Fetch the dataset
        tweet_full = datasets.load_dataset("tweet_eval", "sentiment")

        self.tweet_train = self.tokenize_data(tweet_full["train"], self.tokenizer)
        self.tweet_val = self.tokenize_data(tweet_full["validation"], self.tokenizer)
        self.tweet_test = self.tokenize_data(tweet_full["test"], self.tokenizer)

    def setup(self, stage: Optional[str]) -> None:
        if stage == "predict":
            self.tweet_pred = datasets.concatenate_datasets(
                [self.tweet_val, self.tweet_test]
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.tweet_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_pred,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    @staticmethod
    def tokenize_data(dataset: Dataset, tokenizer: BertTokenizer) -> Dataset:
        return dataset.map(
            lambda e: tokenizer(
                e["text"], truncation=True, padding="max_length", max_length=120
            ),
            batched=True,
        )

    @staticmethod
    def tweeteval_batch(batch: List) -> Tuple[Dict[str, Any], List[Any]]:
        samples = {
            k: torch.tensor([dic[k] for dic in batch]) for k in batch[0] if k != "text"
        }
        labels = samples.pop("label")

        return samples, labels
