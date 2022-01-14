import os
import torch
import argparse

from models import model_picker
from tools.utils import get_checkpoint_hparams


def export(args: argparse.Namespace) -> str:
    """Export the Pytorch Lightning model/checkpoint to ONNX.
    Args:
        args (argparse.Namespace): checkpoint, desired output file, etc...
    """
    if os.path.exists(f"exports/{args.type.lower()}") is False:
        os.makedirs(f"exports/", exist_ok=True)

    model, check_path, _ = get_checkpoint_hparams(args.checkpoint)
    tweesent = model_picker(model).load_from_checkpoint(check_path)

    sample = {
        "attention_mask": torch.randint(2, (1, 6)),
        "input_ids": torch.randint(20, (1, 6)),
    }
    tweesent.to_onnx(
        f"exports/{args.type.lower()}/{args.name}.onnx",
        input_sample=(sample["attention_mask"], sample["input_ids"]),
        input_names=["input_ids", "attention_mask"],
        output_names=["model_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "model_output": {0: "batch_size"},
        },
        do_constant_folding=True,
        opset_version=12,
    )

    return f"Exported {args.name}!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--name", type=str, help="Name of the ONNX / Torchscript file")

    args = parser.parse_args()
    export(args)
