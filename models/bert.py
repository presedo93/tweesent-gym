import torch
from transformers import BertModel

from models.core import CoreTweeSent

BERT_DESC = "Bert Model with a Dropout layer and a Linear layer as output..."


class BertSentiment(CoreTweeSent):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.desc = BERT_DESC

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.freeze_bert()

    def freeze_bert(self):
        if isinstance(self.bert, BertModel):
            self.bert.eval()
            self.bert.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        attention_mask: torch.LongTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        _, pooled_output = self.bert(
            attention_mask=attention_mask,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        ).values()
        output = self.drop(pooled_output)
        return self.out(output)
