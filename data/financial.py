import os
import torch
import datasets
import pytorch_lightning as pl

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import DataCollatorWithPadding
from typing import Optional, Any, Dict, List, Tuple, Union

FINAN_DESC = (
    "Dataset that brings the sentiment over 5000 sentences about financial topics"
)


class FinancialModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer: str, *, workers: int = 4, batch_size: int = 16, **kwargs
    ) -> None:
        super().__init__()
        self.desc = FINAN_DESC

        # Check if tokenizer parallelism is enabled.
        if os.getenv("TOKENIZERS_PARALLELISM") in [None, "false"]:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Train/test/ stage parameters
        self.batch_size = batch_size
        self.workers = workers

        # Tokenizer from pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def prepare_data(self) -> None:
        finan_full = datasets.load_datasets("financial_phrasebank", "sentences_50agree")

        # Split full dataset for train and test datasets
        finan_train_test = finan_full["train"].train_test_split(
            test_size=0.2, shuffle=True
        )

        # Split train dataset in train and validation datasets
        finan_train_val = finan_train_test["train"].train_test_split(
            test_size=0.1, shuffle=True
        )

        self.finan_train = self.tokenize_data(finan_train_val["train"], self.tokenizer)
        self.finan_val = self.tokenize_data(finan_train_val["test"], self.tokenizer)
        self.finan_test = self.tokenize_data(finan_train_test["test"], self.tokenizer)

    def setup(self, stage: Optional[str]) -> None:
        if stage == "predict":
            self.finan_pred = datasets.concatenate_datasets(
                [self.finan_val, self.finan_test]
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.tweet_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=self.financial_batch,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.financial_batch,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.financial_batch,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.tweet_pred,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.financial_batch,
        )

    def financial_batch(self, batch: List) -> Tuple[Dict[str, Any], List[Any]]:
        samples = {k: [dic[k] for dic in batch] for k in batch[0] if k != "sentence"}
        labels = torch.tensor(samples.pop("label"))
        samples = self.data_collator(samples)

        return samples, labels

    @staticmethod
    def tokenize_data(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        return dataset.map(
            lambda e: tokenizer(e["sentence"], truncation=True), batched=True
        )
