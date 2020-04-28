import torch
from criterion import *

def train(model, d, optimizer, device):

        model.train()
        
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        targets_start = d["targets_start"].to(device, dtype=torch.long)
        targets_end = d["targets_end"].to(device, dtype=torch.long)

        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)
        
        loss1 = loss_ce(outputs_start, targets_start)
        loss2 = loss_ce(outputs_end, targets_end)
        
        loss = 0.5*loss1 + 0.5*loss2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()
