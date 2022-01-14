import argparse
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer

LABELS = ["negative", "neutral", "positive"]


def inference(args: argparse.Namespace) -> float:
    """Inference an ONNX model exported from Pytorch
    Lightning.

    Args:
        args (argparse.Namespace): file, mode, engine, etc...
    """

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sample = tokenizer(args.words, truncation=True)

    tweesent = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider"])

    attention_mask = tweesent.get_inputs()[0].name
    input_ids = tweesent.get_inputs()[1].name

    f_inputs = {
        attention_mask: np.expand_dims(sample["attention_mask"], axis=0),
        input_ids: np.expand_dims(sample["input_ids"], axis=0),
    }
    out = tweesent.run(None, f_inputs)
    print("Prediction:", LABELS[out[0].argmax()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, help="Inputs: File or checkpoint")
    parser.add_argument("--words", type=str, help="Path to the CSV data")

    args = parser.parse_args()
    inference(args)
