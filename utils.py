import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def ATLOP_collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output

def SimCSE_collecte_fn(batch):
    max_len = max([len(f['origin']) for f in batch])
    input_ids=[]
    input_masks=[]
    for feature in batch:
        input_ids.append([value + [0] * (max_len - len(value)) for value in feature.values()])
        input_masks.append([[1.0] * len(value) + [0.0] * (max_len - len(value)) for value in feature.values()])
        # input_mask = []
        # for value in input_id:
        #     input_mask.append([1.0 if i>0 else 0.0 for i in value])
        # input_masks.append(input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.float)
    return (input_ids, input_masks)
