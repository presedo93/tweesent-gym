from typing import Any, Dict, KeysView
from models.core import CoreTweeSent
from models.bert import BertSentiment, BERT_DESC
from models.distilbert import DistilBertSentiment, DISTILBERT_DESC


MODELS: Dict[str, Dict[str, Any]] = {
    "bert": {"model": BertSentiment, "desc": BERT_DESC},
    "distilbert": {"model": DistilBertSentiment, "desc": DISTILBERT_DESC},
}


def available_models() -> KeysView[str]:
    return MODELS.keys()


def model_picker(model_name: str) -> CoreTweeSent:
    try:
        return MODELS[model_name.lower()]["model"]
    except KeyError:
        return MODELS["core"]["model"]


def desc_model(model_name: str) -> str:
    try:
        return MODELS[model_name.lower()]["desc"]
    except KeyError:
        return "Model has no description"
