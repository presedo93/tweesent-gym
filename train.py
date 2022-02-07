import argparse
import pytorch_lightning as pl

from typing import Dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from models import model_picker
from tools.progress import StProgressBar
from data.tweeteval import TweetEvalModule
from tools.utils import get_checkpoint_hparams


def train(args: argparse.Namespace, is_st: bool = False) -> Dict:
    """Trains a model in the dataset. It also performs the test stage.

    Args:
        args (argparse.Namespace): parameters and config used to generate
        the dataset and to create/load the model.
        is_st (bool, optional): checks if the method is called from a
        streamlit dashboard. Defaults to False.

    Returns:
        Tuple[fg.Figure, float]: a figure with the predictions and the metric from it.
    """

    if "checkpoint" in args:
        model, check_path, _ = get_checkpoint_hparams(args.checkpoint)
        tweesent = model_picker(model).load_from_checkpoint(check_path)
    else:
        tweesent = model_picker(args.model)(**vars(args))

    # TODO: Support more datasets
    datatext = TweetEvalModule(args.tokenizer)

    # Define the logger used to store the metrics.
    loggers = [
        pl_loggers.TensorBoardLogger(
            "tb_logs/", name=args.model, version=args.dataset, default_hp_metric=False
        )
    ]
    if "wandb" in args.loggers.lower():
        loggers += [pl_loggers.WandbLogger(save_dir="tb_logs/", name=args.model)]

    # Set the callbacks used during the stages.
    callbacks = [EarlyStopping("loss/valid", patience=6)]
    callbacks += [LearningRateMonitor(logging_interval="epoch")]
    if is_st:
        callbacks += [StProgressBar()]

    # Create the trainer with the params.
    trainer = pl.Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)

    # Find the optimal learning rate.
    if args.auto_lr_find:
        trainer.tune(tweesent, datamodule=datatext)

    # Start the training/validation/test process.
    trainer.fit(tweesent, datatext)
    trainer.test(datamodule=datatext)

    metrics = tweesent.get_metrics(["all"])
    if not is_st:
        print(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--model", type=str, default="distilbert", help="Model to train on"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tweet_eval",
        help="Dataset from Hugging Face Hub used",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-cased",
        help="Tokenizer for the text",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument(
        "--metrics", type=str, help="TorchMetrics to evaluate the model"
    )
    parser.add_argument(
        "--loggers",
        type=str,
        default="WandB",
        help="Loggers to use: WandB, TensorBoard",
    )

    # Training type params
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning Rate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="Num of workers for dataloaders"
    )

    # Enable pytorch lightning trainer arguments from cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
