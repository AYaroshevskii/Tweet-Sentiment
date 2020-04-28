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
parser.add_argument("--model", type=str, default="RoBerta")
parser.add_argument("--model_type", type=str, default='small')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--num_folds", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=6)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--warmup", type=int, default=300)
parser.add_argument("--start_decay", type=int, default=1)
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--steps_eval", type=int, default=50)
parser.add_argument("--pseudo_labels", type=bool, default=False)
args = parser.parse_args()

print (args)

# Argument parser
MODEL_NAME=args.model 
MODEL_TYPE=args.model_type

START_EPOCH=args.start_epoch
NUM_EPOCH=args.max_epoch
BATCH_SIZE=args.batch_size

FOLD=args.fold
NUM_FOLDS=args.num_folds
SEED=args.seed

WARMUP=args.warmup
START_DECAY=args.start_decay
LR_GAMMA=args.lr_gamma
STEPS_EVAL=args.steps_eval

PSEUDO_LABELS=args.pseudo_labels

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

# KFOLD split
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for idx_fold, (train_index, test_index) in enumerate(skf.split(df, df.sentiment)):
    if idx_fold==FOLD: break
        
valid_df = df.iloc[test_index]
df = df.iloc[train_index]

if (PSEUDO_LABELS):
    df = pd.concat([df, pd.read_csv('data/PseudoLabels.csv')], axis=0, sort=True)


# Create dataset objects
train_dataset = TweetDataset(
        tweet=df.text.values,
        sentiment=df.sentiment.values,
        selected_text=df.selected_text.values
    )

valid_dataset = TweetDataset(
        tweet=valid_df.text.values,
        sentiment=valid_df.sentiment.values,
        selected_text=valid_df.selected_text.values
    )

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4,
                               worker_init_fn=_worker_init_fn)

valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4,
                               worker_init_fn=_worker_init_fn)

# Create Model
device = torch.device(args.device)

if (MODEL_NAME == 'RoBerta'):
    model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
    
    if (MODEL_TYPE == 'small'):
        model_config.output_hidden_states = True

    else: #LARGE
        model_config.num_attention_heads = 16
        model_config.num_hidden_layers = 24
        model_config.intermediate_size = 4096
        model_config.hidden_size = 1024
        model_config.output_hidden_states = True
    
    model = RobertaBase(conf=model_config).to(device)

# Set optimizers
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = 2e-5,
                              eps = 1e-8)

scheduler = get_constant_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=WARMUP)

scheduler_decay = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=1,
                                                  gamma=LR_GAMMA)

# Fine-tuning with pseudo-labels
if (PSEUDO_LABELS):
    load_checkpoint('Models/model{}.pth'.format(FOLD), model, None)
    optimizer = torch.optim.AdamW(model.parameters(),
                              lr = 1e-6,
                              eps = 1e-8)
    START_DECAY = NUM_EPOCH+1
    START_EPOCH = 1
    NUM_EPOCH = START_EPOCH+2

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
                save_checkpoint('Models/{}model{}.pth'.format(MODEL_TYPE, FOLD), model, optimizer)

            print ("EPOCH: {}, STEP: {}".format(epoch, step))
            print("Train Loss : {}, Validation Jac Score : {}".format(
                  np.mean(train_loss), mean_jac_score))
    

print ("Best Valid Jaccard Score: ", TOTAL_SCORE)
