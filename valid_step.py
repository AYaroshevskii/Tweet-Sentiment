import torch
from tqdm import tqdm
import numpy as np
from criterion import *


def valid(model, valid_data_loader, device):

    mean_jac_score = 0
    model.eval()

    for val_d in tqdm(valid_data_loader, total=len(valid_data_loader)):

        ids = val_d["ids"].to(device, dtype=torch.long)
        token_type_ids = val_d["token_type_ids"].to(device, dtype=torch.long)
        mask = val_d["mask"].to(device, dtype=torch.long)
        sentiment = val_d["sentiment"].numpy()
        orig_selected = val_d["orig_selected"]
        orig_tweet = val_d["orig_tweet"]
        offsets = val_d["offsets"].numpy()

        with torch.no_grad():
            outputs_start, outputs_end, _ = model(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]

            # prob_matrix = 4 * outputs_start[px].unsqueeze(-1).repeat(1, outputs_start[px].size(-1))
            
            # prob_matrix += outputs_end[px]
            # prob_matrix = prob_matrix.triu(diagonal=1).cpu().detach().numpy()

            # start_idx, end_idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
            #end_idx = prob_matrix.argmax(dim=1)

            #print(start_idx, end_idx)

            predict_selected = get_selected_string(
                original_tweet=tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, np.argmax(outputs_start[px, :]):]) + np.argmax(outputs_start[px, :]),
                offsets=offsets[px])
            #print(selected_tweet, predict_selected)

            mean_jac_score += jaccard(predict_selected, selected_tweet)

    return mean_jac_score
