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

    pos_input_ids = [f["pos_input_ids"] + [0] * (max_len - len(f["pos_input_ids"])) for f in batch if "pos_input_ids" in f.keys()]
    pos_input_mask = [[1.0] * len(f["pos_input_ids"]) + [0.0] * (max_len - len(f["pos_input_ids"])) for f in batch if "pos_input_ids" in f.keys()]
    tmp_eids = [f["eids_map"] for f in batch if "eids_map" in f.keys()]
    evidence_entity_pos = [f["evidence_entity_pos"] for f in batch if "evidence_entity_pos" in f.keys()]
    evi_hts = [f["evi_hts"] for f in batch if "evi_hts" in f.keys()]

    hts_map = []
    for i, f in enumerate(batch):
        if "hts_map" in f.keys():
            if i > 0:
                hts_map.extend([ht + len(hts[i-1]) for ht in f['hts_map']])
            else: 
                hts_map.extend([ht for ht in f['hts_map']])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    pos_input_ids = torch.tensor(pos_input_ids, dtype=torch.long)
    pos_input_mask = torch.tensor(pos_input_mask, dtype=torch.float)

    output = (input_ids, input_mask, labels, entity_pos, hts, pos_input_ids, pos_input_mask, tmp_eids, evidence_entity_pos, evi_hts, hts_map)
    return output
