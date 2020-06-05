from models import *
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
parser.add_argument("--model_name", type=str, default='roberta-base')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--num_folds", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=4)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--warmup", type=int, default=300)
parser.add_argument('--start_lr', type=int, default=2)
parser.add_argument("--start_decay", type=int, default=1)
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--steps_eval", type=int, default=5)
parser.add_argument("--pseudo_labels", type=bool, default=True)
parser.add_argument("--frac", type=float, default=0.5)
args = parser.parse_args()

print (args)

# Argument parser
MODEL_NAME=args.model_name

START_EPOCH=args.start_epoch
NUM_EPOCH=args.max_epoch
BATCH_SIZE=args.batch_size

FOLD=args.fold
NUM_FOLDS=args.num_folds
SEED=args.seed

START_LR=args.start_lr
WARMUP=args.warmup
START_DECAY=args.start_decay
LR_GAMMA=args.lr_gamma
STEPS_EVAL=args.steps_eval

PSEUDO_LABELS=args.pseudo_labels
PS_FRACTION=args.frac
TOTAL_SCORE=0

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
    add_prefix_space=True)

# KFOLD split
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for idx_fold, (train_index, test_index) in enumerate(skf.split(df, df.sentiment)):
    if idx_fold==FOLD: break

valid_df = df.iloc[test_index]
df = df.iloc[train_index]

if (PSEUDO_LABELS):
    ps_labels = pd.read_csv('data/PSLabels.csv')
    for idx_fold, (train_index, test_index) in enumerate(skf.split(ps_labels, ps_labels.sentiment)):
        if idx_fold==FOLD: break

    ps_labels = ps_labels.iloc[test_index]
    ps_labels = ps_labels.sample(frac=PS_FRACTION, random_state=SEED)
    df = pd.concat([df, ps_labels], axis=0, sort=True)


# Create dataset objects
train_dataset = TweetDataset(
        tweet=df.text.values,
        sentiment=df.sentiment.values,
        selected_text=df.selected_text.values,
        tokenizer=TOKENIZER)

valid_dataset = TweetDataset(
        tweet=valid_df.text.values,
        sentiment=valid_df.sentiment.values,
        selected_text=valid_df.selected_text.values,
        tokenizer=TOKENIZER)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4,
                               worker_init_fn=_worker_init_fn)

valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4,
                               worker_init_fn=_worker_init_fn)

# Create Model
device = torch.device(args.device)

model_config = CONFIG[MODEL_NAME](MODEL_NAME)

model = ModelBase(conf=model_config, model_name=MODEL_NAME).to(device)

# Set optimizers
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = START_LR*1e-5,
                              eps = 1e-8)

scheduler = get_constant_schedule_with_warmup(optimizer,
                                              num_warmup_steps=WARMUP)

scheduler_decay = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=1,
                                                  gamma=LR_GAMMA)

# Train
for epoch in range(START_EPOCH, NUM_EPOCH):

    train_loss = []

    if (epoch >= START_DECAY): scheduler_decay.step() # LR DECAY 

    for step, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

        loss = train(model, d, optimizer, device) # TRAIN
        train_loss.append(loss)

        if (epoch == 0): scheduler.step() # WARMUP

        if validation_time(step, STEPS_EVAL, epoch, len(train_data_loader)): # Valid
            
            mean_jac_score = valid(model, valid_data_loader, device)/len(valid_dataset)
        
            if mean_jac_score > TOTAL_SCORE:
                TOTAL_SCORE = mean_jac_score
                save_checkpoint('Models/{}{}.pth'.format(MODEL_NAME, FOLD), model, optimizer)

            print ("EPOCH: {}, STEP: {}".format(epoch, step))
            print("Train Loss : {}, Validation Jac Score : {}".format(
                  np.mean(train_loss), mean_jac_score))
    

print ("Best Valid Jaccard Score: ", TOTAL_SCORE)
