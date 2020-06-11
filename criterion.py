import torch.nn as nn
import torch
import numpy as np

loss_ce = nn.CrossEntropyLoss()
logsoftmax = torch.nn.LogSoftmax(dim=1)

def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), dim=1))

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_selected_string(
    original_tweet,
    sentiment_val,
    idx_start,
    idx_end,
    offsets,
    verbose=False,
):

    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    return filtered_output
