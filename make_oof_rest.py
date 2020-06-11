from models_all import *
from DataLoader import *
from utils import *
from criterion import *
from train_step import *
from valid_step import *
import pandas as pd
import argparse
from torch.autograd import Variable
import random
from transformers import get_constant_schedule_with_warmup
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import string

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--key", type=str, default="")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

print(args)

# Argument parser
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size

NUM_FOLDS = args.num_folds
SEED = args.seed

# SEED BLOCK
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


# TOKENIZER
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"configs/{MODEL_NAME}/vocab.json",
    merges_file=f"configs/{MODEL_NAME}/merges.txt",
    lowercase=True,
    add_prefix_space=True,
)

def find_failed_start(text, selected_text):
    """find the case that lable is not truncated by space"""
    begin = text.find(selected_text)
    end = begin + len(selected_text)
    if begin == 0:
        return False
    return text[begin -1].isalpha()

def find_failed_end(text,selected_text):
    """find the case that lable is not truncated by space"""
    begin = text.find(selected_text)
    end = begin + len(selected_text)
    if end == len(text):
        return False
    return text[end].isalpha()

def spaces_offset(x):
    while x.endswith(' '):
        x = x[:-1]
    
    flag = False
    res = 0
    for ch in x:
        if flag and ch == " ":
            res += 1
        elif flag and ch != " ":
            flag = False
        elif not flag and ch == " ":
            flag = True
    return res

def correct(x):
    idx0 = x.text.find(x.selected_text)
    offset = spaces_offset(x.text[:idx0])
    if offset > 0 and x.text[0] != " ":
        offset -=1
    
    return x.text[max(idx0+offset, 0): idx0+offset+len(x.selected_text)]
df = df.dropna()
#df["selected_text"] = df.apply(lambda x: correct(x), axis=1).str.strip()

failed_start = df.apply(lambda row: find_failed_start(row.text, row.selected_text),axis=1)
failed_end = df.apply(lambda row: find_failed_end(row.text, row.selected_text),axis=1)
df = df[~((~failed_start) & (~failed_end) & (~(df.selected_text == "")))]
print(df.shape)

START = np.zeros((len(df), MAX_LEN))
END = np.zeros((len(df), MAX_LEN))

# KFOLD split
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
df["pred"] = None
orig_tweets_list = []
offsets_list = []
sentiment_list = []

for idx_fold, (train_index, test_index) in enumerate(skf.split(df, df.sentiment)):

    start_pr = []
    end_pr = []
    text_results = []

    FOLD = idx_fold

    valid_df = df#.iloc[test_index]

    valid_dataset = TweetDataset(
        tweet=valid_df.text.values,
        sentiment=valid_df.sentiment.values,
        selected_text=valid_df.selected_text.values,
        tokenizer=TOKENIZER,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_worker_init_fn,
    )

    # Create Model
    device = torch.device(args.device)

    model_config = CONFIG[MODEL_NAME](MODEL_NAME)

    model = ModelBase(conf=model_config, model_name=MODEL_NAME).to(device)

    load_checkpoint("../models/{}{}_f{}.pth".format(args.key, MODEL_NAME, FOLD), model, None)

    model.eval()

    for val_d in tqdm(valid_data_loader, total=len(valid_data_loader)):

        ids = val_d["ids"].to(device, dtype=torch.long)
        token_type_ids = val_d["token_type_ids"].to(device, dtype=torch.long)
        mask = val_d["mask"].to(device, dtype=torch.long)
        orig_tweet = val_d["orig_tweet"]
        offsets = val_d["offsets"].numpy()
        sentiment = val_d["sentiment"].numpy()

        with torch.no_grad():
            outputs_start, outputs_end, _ = model(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        start_pr.extend(outputs_start)
        end_pr.extend(outputs_end)

        orig_tweets_list.extend(orig_tweet)
        offsets_list.extend(offsets)
        sentiment_list.extend(sentiment)
        # for px, tweet in enumerate(orig_tweet):
        #     tweet_sentiment = sentiment[px]

        #     res_text = get_selected_string(
        #         original_tweet=tweet,
        #         sentiment_val=tweet_sentiment,
        #         idx_start=np.argmax(outputs_start[px, :]),
        #         idx_end=np.argmax(outputs_end[px, np.argmax(outputs_start[px, :]):]) + np.argmax(outputs_start[px, :]),
        #         offsets=offsets[px])
        #     text_results.append(res_text)

    START += np.array(start_pr) / 5
    END += np.array(end_pr) / 5

print(START.shape)
text_results = []
for px in range(len(START)):
    #for px, tweet in enumerate(orig_tweets_list):
    tweet_sentiment = sentiment_list[px]
    tweet = orig_tweets_list[px]

    res_text = get_selected_string(
        original_tweet=tweet,
        sentiment_val=tweet_sentiment,
        idx_start=np.argmax(START[px, :]),
        idx_end=np.argmax(END[px, np.argmax(START[px, :]):]) + np.argmax(START[px, :]),
        offsets=offsets_list[px])
    text_results.append(res_text)
    
df["pred"] = text_results

#np.save("../oof/oof_start_{}.npy".format(args.key), START)
#np.save("../oof/oof_end_{}.npy".format(args.key), END)
df.to_csv("../oof/oof_{}_rest.csv".format(args.key), index=None)
