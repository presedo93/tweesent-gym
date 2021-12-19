from typing import Any, Dict, KeysView
from models.core import CoreTweeSent
from models.bert import BertSentiment, BERT_DESC


MODELS: Dict[str, Dict[str, Any]] = {
    "bert": {"model": BertSentiment, "desc": BERT_DESC}
}


def available_models() -> KeysView[str]:
    return MODELS.keys()


# TODO: raise not available model
def model_picker(model_name: str) -> CoreTweeSent:
    return MODELS[model_name.lower()]["model"]


# TODO: raise not available model
def desc_picker(model_name: str) -> str:
    return MODELS[model_name.lower()]["desc"]
