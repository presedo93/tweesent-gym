import argparse
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from data.tweeteval import TweetEvalModule
from models import model_picker


def train(args: argparse.Namespace, progress_bar=None) -> None:
    """Trains a moodel on the dataset. It also performs the test stage and
    the makes predictions in the testset to plot the results.

    Args:
        args (argparse.Namespace): parameters and config used to generate
        the dataset and to create/load the model.
        is_st (bool, optional): checks if the method is called from a
        streamlit dashboard. Defaults to False.

    Returns:
        Tuple[fg.Figure, float]: a figure with the predictions and the metric from it.
    """

    datatext = TweetEvalModule(args.tokenizer)
    tweesent = model_picker(args.model)(**vars(args))

    # Define the logger used to store the metrics.
    tb_logger = pl_loggers.TensorBoardLogger(
        "tb_logs/", name=args.model, version=args.dataset, default_hp_metric=False
    )
    if "wandb" in args.loggers.lower():
        wb_logger = pl_loggers.WandbLogger(save_dir="tb_logs/", name=args.model)
        loggers = [tb_logger, wb_logger]
    else:
        loggers = tb_logger

    # Set the callbacks used during the stages.
    early_stopping = EarlyStopping("loss/valid", patience=12)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    if progress_bar is None:
        progress_bar = RichProgressBar()
    custom_callbacks = [early_stopping, lr_monitor, progress_bar]

    # Create the trainer with the params.
    trainer = pl.Trainer.from_argparse_args(
        args, logger=loggers, callbacks=custom_callbacks, enable_model_summary=False
    )

    # Find the optimal learning rate.
    if args.auto_lr_find:
        trainer.tune(tweesent, datamodule=datatext)

    # Start the training/validation/test process.
    trainer.fit(tweesent, datatext)
    # trainer.test(datamodule=datatext)

    # Evaluate the model with the test set and the val set.
    # trainer.predict(tweesent, datamodule=datatext)

    # metrics = tweesent.get_metrics(["all"])
    # print(metrics)


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
