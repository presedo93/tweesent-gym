import os
import torch
import datasets
import numpy as np
import pytorch_lightning as pl

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List, Optional, Tuple, Union


class TweetEvalModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer: str, *, workers: int = 4, batch_size: int = 16, **kwargs
    ):
        super().__init__()
        # Check if tokenizer parallelism is enabled.
        if os.getenv("TOKENIZERS_PARALLELISM") in [None, "false"]:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Train/test stage parameters.
        self.batch_size = batch_size
        self.workers = workers

        # Tokenizer from pre-trained model.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def prepare_data(self) -> None:
        tweet_full = datasets.load_dataset("tweet_eval", "sentiment")

        # TODO: Only for testing!
        tweet_full["train"] = tweet_full["train"].select(
            np.random.randint(0, 20000, 6000)
        )
        tweet_full["validation"] = tweet_full["validation"].select(
            np.random.randint(0, 2000, 600)
        )
        tweet_full["test"] = tweet_full["test"].select(np.random.randint(0, 2000, 1000))

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
            shuffle=True,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_pred,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.tweeteval_batch,
        )

    def tweeteval_batch(self, batch: List) -> Tuple[Dict[str, Any], List[Any]]:
        samples = {k: [dic[k] for dic in batch] for k in batch[0] if k != "text"}
        labels = torch.tensor(samples.pop("label"))
        samples = self.data_collator(samples)

        return samples, labels

    @staticmethod
    def tokenize_data(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        return dataset.map(
            lambda e: tokenizer(e["text"], truncation=True), batched=True
        )
