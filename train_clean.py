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
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import string

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--num_folds", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=4)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--warmup", type=int, default=300)
parser.add_argument("--start_lr", type=int, default=2)
parser.add_argument("--start_decay", type=int, default=1)
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--steps_eval", type=int, default=100)
parser.add_argument("--pseudo_labels", type=bool, default=False)
parser.add_argument("--frac", type=float, default=0.5)
parser.add_argument("--inner_k", type=float, default=0.02)
parser.add_argument("--key", type=str, default="")
args = parser.parse_args()

print(args)

# Argument parser
MODEL_NAME = args.model_name

START_EPOCH = args.start_epoch
NUM_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size

FOLD = args.fold
NUM_FOLDS = args.num_folds
SEED = args.seed

START_LR = args.start_lr
WARMUP = args.warmup
START_DECAY = args.start_decay
LR_GAMMA = args.lr_gamma
STEPS_EVAL = args.steps_eval

PSEUDO_LABELS = args.pseudo_labels
PS_FRACTION = args.frac
TOTAL_SCORE = 0

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

# KFOLD split
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)



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
df = df[(~failed_start) & (~failed_end) & (~(df.selected_text == ""))]

for idx_fold, (train_index, test_index) in enumerate(skf.split(df, df.sentiment)):
    if idx_fold == FOLD:
        break

valid_df = df.iloc[test_index]
train_df = df.iloc[train_index].dropna()


if PSEUDO_LABELS:
    ps_labels = pd.read_csv("../input/PSLabels.csv")
    for idx_fold, (train_index, test_index) in enumerate(
        skf.split(ps_labels, ps_labels.sentiment)
    ):
        if idx_fold == FOLD:
            break

    ps_labels = ps_labels.iloc[test_index]
    ps_labels = ps_labels.sample(frac=PS_FRACTION, random_state=SEED)
    df = pd.concat([df, ps_labels], axis=0, sort=True)


# Create dataset objects
train_dataset = TweetDataset(
    tweet=train_df.text.values,
    sentiment=train_df.sentiment.values,
    selected_text=train_df.selected_text.values,
    tokenizer=TOKENIZER,
)

valid_dataset = TweetDataset(
    tweet=valid_df.text.values,
    sentiment=valid_df.sentiment.values,
    selected_text=valid_df.selected_text.values,
    tokenizer=TOKENIZER,
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    worker_init_fn=_worker_init_fn,
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

# Set optimizers
params = list(model.named_parameters())
# Set optimizers
def is_backbone(n):
        return "backbone" in n

optimizer_grouped_parameters = [
    {'params': [p for n, p in params if is_backbone(n)],
        'lr': START_LR * 1e-5},
    {'params': [p for n, p in params if not is_backbone(n)],
        'lr': START_LR * 1e-5 * 500}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=START_LR * 1e-5, weight_decay=0)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=WARMUP,
    num_training_steps=688*4)

scheduler_decay = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=LR_GAMMA
)

# Train
for epoch in range(START_EPOCH, NUM_EPOCH):

    train_loss = []

    # if epoch >= START_DECAY:
    #     scheduler_decay.step()  # LR DECAY

    for step, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

        loss = train(model, d, optimizer, device, args.inner_k)  # TRAIN
        train_loss.append(loss)

        #if epoch == 0:
        scheduler.step()  # WARMUP

        if validation_time(step, STEPS_EVAL, epoch, len(train_data_loader)):  # Valid

            mean_jac_score = valid(model, valid_data_loader, device) / len(
                valid_dataset
            )

            if mean_jac_score > TOTAL_SCORE:
                TOTAL_SCORE = mean_jac_score
                save_checkpoint(
                    "../models/{}{}_f{}.pth".format(args.key, MODEL_NAME, FOLD), model, optimizer
                )

            print("EPOCH: {}, STEP: {}".format(epoch, step))
            print(
                "Train Loss : {}, Validation Jac Score : {}".format(
                    np.mean(train_loss), mean_jac_score
                )
            )


print("Best Valid Jaccard Score: ", TOTAL_SCORE)
