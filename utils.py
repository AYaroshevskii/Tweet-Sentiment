from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm_notebook
from torch.utils import data
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import torch
import transformers

ROBERTA_PATH = "roberta-base"

# Save and load model checkpoint
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {"state_dict": model.state_dict()}
    if optimizer:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, checkpoint_path)
    print("model saved to %s" % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
    print("model loaded from %s" % checkpoint_path)

def validation_time(step, steps_eval, epoch, n_valid):
    return  (step%steps_eval == 0 and epoch > 1 and step > 0) or (step+1==n_valid)

def sentiment_to_class(s):
    if s == 'neutral':
        return 1
    if s == 'negative':
        return 0
    if s == 'positive':
        return 2

CONFIG = {'roberta-base': transformers.RobertaConfig.from_pretrained,
           'roberta-large': transformers.RobertaConfig.from_pretrained,
           'distilroberta-base': transformers.RobertaConfig.from_pretrained,
           "roberta-base-openai-detector": transformers.RobertaConfig.from_pretrained,
           "gpt2": transformers.GPT2Config.from_pretrained,
           "gpt2-medium": transformers.GPT2Config.from_pretrained}

MODELS = {'roberta-base': transformers.RobertaModel.from_pretrained,
          'roberta-large': transformers.RobertaModel.from_pretrained,
          'distilroberta-base': transformers.RobertaModel.from_pretrained,
          "roberta-base-openai-detector": transformers.RobertaModel.from_pretrained,
          "gpt2": transformers.GPT2Model.from_pretrained,
          "gpt2-medium": transformers.GPT2Model.from_pretrained}
