import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import transformers
from utils import *


class ModelBase(transformers.BertPreTrainedModel):
    def __init__(self, conf, model_name):
        super(ModelBase, self).__init__(conf)

        self.backbone = MODELS[model_name](model_name, output_hidden_states=True)
        
        self.drop_out = nn.Dropout(0.5)

        self.l0 = nn.Linear(conf.hidden_size * 2, 2)
        
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.backbone(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        
        logits = torch.mean(
            torch.stack(
                [self.l0(self.drop_out(out)) for _ in range(5)],
                dim=0,),
                dim=0,)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
