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
parser.add_argument("--model_name", type=str, default='distilroberta-base')
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

print (args)

# Argument parser
MODEL_NAME=args.model_name
BATCH_SIZE=args.batch_size

NUM_FOLDS=args.num_folds
SEED=args.seed

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


START=np.zeros((len(df), MAX_LEN))
END=np.zeros((len(df), MAX_LEN))

# KFOLD split
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for idx_fold, (train_index, test_index) in enumerate(skf.split(df, df.sentiment)):

    start_pr = []
    end_pr = []

    FOLD=idx_fold
        
    valid_df = df.iloc[test_index]

    valid_dataset = TweetDataset(
        tweet=valid_df.text.values,
        sentiment=valid_df.sentiment.values,
        selected_text=valid_df.selected_text.values,
        tokenizer=TOKENIZER)

    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4,
                               worker_init_fn=_worker_init_fn)

    # Create Model
    device = torch.device(args.device)

    model_config = CONFIG[MODEL_NAME](MODEL_NAME)

    model = ModelBase(conf=model_config, model_name=MODEL_NAME).to(device)

    load_checkpoint('Models/ps{}{}.pth'.format(MODEL_NAME, FOLD), model, None)

    model.eval()

    for val_d in tqdm(valid_data_loader, total=len(valid_data_loader)):

            ids = val_d["ids"].to(device, dtype=torch.long)
            token_type_ids = val_d["token_type_ids"].to(device, dtype=torch.long)
            mask = val_d["mask"].to(device, dtype=torch.long)

            with torch.no_grad():
                outputs_start, outputs_end = model(ids=ids,
                                                   mask=mask,
                                                   token_type_ids=token_type_ids)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            start_pr.extend(outputs_start)
            end_pr.extend(outputs_end)

    START[test_index]=np.array(start_pr)
    END[test_index]=np.array(end_pr)

np.save('OOF/oof_start_{}.npy'.format(MODEL_NAME), START)
np.save('OOF/oof_end_{}.npy'.format(MODEL_NAME), END)
