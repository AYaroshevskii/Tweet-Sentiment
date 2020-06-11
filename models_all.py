import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from utils import *


class ModelBase(transformers.BertPreTrainedModel):
    def __init__(self, conf, model_name):
        super(ModelBase, self).__init__(conf)

        self.backbone = MODELS[model_name](model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(p=0.2)
        self.drop_out = nn.Dropout(0.5)

        self.l0 = nn.Linear(conf.hidden_size, 3)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

        n_weights = conf.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

    def forward(self, ids, mask, token_type_ids):
        _, _, hidden_layers = self.backbone(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        #cls_output = torch.cat((hidden_layers[-1], hidden_layers[-3]), dim=-1)

        cls_outputs = torch.stack(
            [self.dropout(layer[:, :, :]) for layer in hidden_layers],
            dim=3
        )
        #print(cls_outputs.shape)
        cls_output = (
            torch.softmax(self.layer_weights, dim=0) * cls_outputs
        ).sum(-1)
        # print(cls_output.shape)
        logits = torch.mean(
            torch.stack([self.l0(self.drop_out(cls_output)) for _ in range(5)], dim=0,),
            dim=0,
        )

        # print(logits.shape)
        start_logits, end_logits, ner_logits = logits.split(1, dim=-1)
        # print(start_logits.shape, end_logits.shape)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # print(start_logits, end_logits)

        # print(start_logits.shape, end_logits.shape)

        return start_logits, end_logits, ner_logits.squeeze(-1)
