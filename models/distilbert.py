import rich
import torch
from transformers import DistilBertModel

from models.core import CoreTweeSent

DISTILBERT_DESC = (
    "DistilBert Model with a Dropout layer and two Linear layers as output..."
)


class DistilBertSentiment(CoreTweeSent):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.desc = DISTILBERT_DESC

        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.drop = torch.nn.Dropout(self.distilbert.config.seq_classif_dropout)

        distilbert_dim = self.distilbert.config.dim
        self.pre_classifier = torch.nn.Linear(distilbert_dim, distilbert_dim)
        self.classifier = torch.nn.Linear(distilbert_dim, 3)
        # self.freeze_distilbert()

    def freeze_distilbert(self):
        if isinstance(self.distilbert, DistilBertModel):
            self.distilbert.eval()
            self.distilbert.requires_grad = False
            for param in self.distilbert.parameters():
                param.requires_grad = False

    def forward(
        self, attention_mask: torch.IntTensor, input_ids: torch.IntTensor, **kwargs
    ) -> torch.Tensor:
        output = self.distilbert(attention_mask=attention_mask, input_ids=input_ids)

        hidden_state = output[0]  # (bs, seq_len, dim)
        output = hidden_state[:, 0]  # (bs, dim)

        output = torch.relu(self.pre_classifier(output))
        output = self.drop(output)
        return self.classifier(output)
