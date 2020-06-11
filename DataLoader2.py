from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tokenizers
import re
from utils import *
import random

df = pd.read_csv("../input/train.csv")
df.loc[:, "sentiment"] = df.sentiment.apply(lambda x: sentiment_to_class(x))

MAX_LEN = 128

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def process_data(tweet, selected_text, sentiment, tokenizer, max_len, train=True):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind : ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    # if np.random.rand() < 0.4 and sentiment != 0 and train:
    #     spaces = find(tweet[:idx0], " ")

    #     if len(spaces) > 0:
    #         pos = random.choice(spaces)
    #         #print("before", tweet)
            
    #         nums = np.random.randint(4)
    #         tweet = tweet[:pos] + (" ")*nums + tweet[pos:]
    #         idx0 -= nums
    #         idx1 -= nums

            ##print("after", tweet)
            #print(selected_text, " > ", tweet[max(0,idx0):idx1+2])


    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    else:
        return None

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1:offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = [2430, 7974, 1313]

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    ner = np.zeros(max_len)
    ner[targets_start + 1: targets_end] = 1

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    return {
        "ids": input_ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "targets_start": targets_start,
        "targets_end": targets_end,
        "orig_tweet": tweet,
        "orig_selected": selected_text,
        "sentiment": sentiment,
        "offsets": tweet_offsets,
        "ner": ner,
    }


def process_data_test(tweet, selected_text, sentiment, tokenizer, max_len, train=True):
    tweet = " " + str(tweet)

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets

    sentiment_id = [2430, 7974, 1313]

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    return {
        "ids": input_ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "targets_start": 0,
        "targets_end": 0,
        "orig_tweet": tweet,
        "orig_selected": [],
        "sentiment": sentiment,
        "offsets": tweet_offsets,
        "ner": [],
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tokenizer, train=True):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
        self.train = train

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len,
            self.train
        )

        return {
            "ids": torch.tensor(data["ids"], dtype=torch.long),
            "mask": torch.tensor(data["mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(data["token_type_ids"], dtype=torch.long),
            "targets_start": torch.tensor(data["targets_start"], dtype=torch.long),
            "targets_end": torch.tensor(data["targets_end"], dtype=torch.long),
            "orig_tweet": data["orig_tweet"],
            "orig_selected": data["orig_selected"],
            "sentiment": data["sentiment"],
            "offsets": torch.tensor(data["offsets"], dtype=torch.long),
            "ner": torch.tensor(data["ner"], dtype=torch.float),
        }
