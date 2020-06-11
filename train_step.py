import torch
from criterion import *


def train(model, d, optimizer, device, inner_loss):

    model.train()

    ids = d["ids"].to(device, dtype=torch.long)
    token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
    mask = d["mask"].to(device, dtype=torch.long)
    targets_start = d["targets_start"].to(device, dtype=torch.long)
    targets_end = d["targets_end"].to(device, dtype=torch.long)
    # targets_matrix = torch.zeros(192, 192).cuda()
    # targets_matrix[targets_start, :] = 0.5 / 192 / 2
    # targets_matrix[:, targets_end] = 0.5 / 192 / 2
    # targets_matrix[targets_start, targets_end] = 0.5
    # print(targets_start, targets_end)

    outputs_start, outputs_end, output_ner = model(
        ids=ids, mask=mask, token_type_ids=token_type_ids
    )
    # prob_matrix = outputs_start.unsqueeze(-1).repeat(1, 1, outputs_start.size(-1))
    # #print(prob_matrix.shape, outputs_end[px].shape)
    # prob_matrix += outputs_end.unsqueeze(1)
    # prob_matrix = prob_matrix.triu(diagonal=1)

    #start_idx, end_idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)

    loss1 = loss_ce(outputs_start, targets_start)
    loss2 = loss_ce(outputs_end, targets_end)
    loss3 = cross_entropy(output_ner, d["ner"].to(device))

    loss = 0.5 * loss1 + 0.5 * loss2 + inner_loss * loss3

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
