import os

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from model import DocREModel
from utils import set_seed, ATLOP_collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb

# some bascal arguments
@dataclass
class CommonArguments():
    transformer_type: Optional[str] = field(
        default="bert"
    )
    model_name_or_path: Optional[str] = field(
        default="bert-base-cased"
    )
    config_name: Optional[str] = field(
        default="",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        }
    )
    data_dir: Optional[str] = field(
        default="/data/wh/dataset/docred"
    )
    train_file: Optional[str] = field(
        default="train_annotated.json"
    )
    dev_file: Optional[str] = field(
        default="dev.json"
    )
    test_file: Optional[str] = field(
        default="test.json"
    )
    save_path: Optional[str] = field(
        default=""
    )
    load_path: Optional[str] = field(
        default=""
    )
    load_pretrained: Optional[str] = field(
        default=""
    )

# arguments of ATLOP
@dataclass
class ATLOPArguments():
    axial_attention: Optional[str] = field(
        default="multi_external",
        metadata={
            "choices": ['none', 'self_attention', 'single_external', 'multi_external'],
            "help": "type of AxialAttention"
        }
    )
    tag: Optional[str] = field(
        default="Train",
        metadata={
            "choices": ['Train', 'Dev']
        }
    )
    train_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch size for training."
        }
    )
    test_batch_size: Optional[int] = field(
        default=8,
        metadata={
            "help": "Batch size for testing."
        }
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        }
    )
    num_labels: Optional[int] = field(
        default=4,
        metadata={
            "help": "Max number of labels in prediction."
        }
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": "The initial learning rate for Adam."
        }
    )
    classifier_lr: Optional[float] = field(
        default=1e-4,
        metadata={
            "help": "The initial learning rate for Classifier."
        }
    )
    adam_epsilon: Optional[float] = field(
        default=1e-6,
        metadata={
            "help": "Epsilon for Adam optimizer."
        }
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Max gradient norm."
        }
    )
    warmup_ratio: Optional[float] = field(
        default=0.06,
        metadata={
            "help": "Warm up ratio for Adam."
        }
    )
    num_train_epochs: Optional[float] = field(
        default=30.0,
        metadata={
            "help": "Total number of training epochs to perform."
        }
    )
    evaluation_steps: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of training steps between evaluations."
        }
    )
    seed: Optional[int] = field(
        default=66,
        metadata={
            "help": "random seed for initialization"
        }
    )
    num_class: Optional[int] = field(
        default=97,
        metadata={
            "help": "Number of relation types in dataset."
        }
    )

def get_lr(optimizer):
    lm_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    return lm_lr, classifier_lr

def ATLOP_train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=ATLOP_collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4]
                          }
                if args.tag == "Train":
                    inputs['pos_input_ids'] = batch[5].to(args.device)
                    inputs['pos_input_mask'] = batch[6].to(args.device)
                    inputs['tmp_eids'] = batch[7]
                    inputs['evidence_entity_pos'] = batch[8]
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = ATLOP_evaluate(args, model, dev_features, tag="dev")

                    lm_lr, classifier_lr = get_lr(optimizer)
                    wandb.log(dev_output, step=num_steps)
                    wandb.log({"LM lr" : round(lm_lr,5), "Classifier lr" : round(classifier_lr,5)}, step=num_steps)

                    if dev_score > best_score:
                        best_score = dev_score
                        pred = ATLOP_report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)


        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)

def ATLOP_evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=ATLOP_collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)

    #if len(ans) > 0:
    if tag=='dev':
        best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(ans, args.data_dir, args.train_file, args.dev_file)
    elif tag=='train':
        best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(ans, args.data_dir, args.train_file, args.train_file)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_P": best_p * 100,
        tag + "_R": best_r * 100,
    }
    return best_f1, output

def ATLOP_report(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=ATLOP_collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds

def main():
    parser = HfArgumentParser((CommonArguments, ATLOPArguments))
    args, ATLOP_args= parser.parse_args_into_dataclasses()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device, ATLOP_args.device = device, device
    args.n_gpu, ATLOP_args.n_gpu = torch.cuda.device_count(), torch.cuda.device_count()
    ATLOP_args.data_dir = args.data_dir

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=ATLOPArguments.num_class,
    )
    # Bert Model
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    
    read = read_docred
    suffix = '.{}.pt'.format(args.model_name_or_path)
    # train_features
    if os.path.exists(os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix)):
        train_features = torch.load(os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length, axial_attention=ATLOP_args.axial_attention, tag=ATLOP_args.tag)
        torch.save(train_features, os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix))
        print('Created and saved new train features')
    # dev_features   
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix)):
        dev_features= torch.load(os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, axial_attention=ATLOP_args.axial_attention, tag=ATLOP_args.tag)
        torch.save(dev_features, os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix))
        print('Created and saved new dev features')
    # test_features
    if os.path.exists(os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix)):
        test_features = torch.load(os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, axial_attention=ATLOP_args.axial_attention)
        torch.save(test_features, os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix))
        print('Created and saved new test features')

    #Training
    wandb.init(project="ATLOP-TEST", entity="15346186000", config=ATLOP_args)
    
    set_seed(ATLOP_args)
    model = DocREModel(config, model, num_labels=ATLOP_args.num_labels, axial_attention=ATLOP_args.axial_attention)
    model.to(device)

    if args.load_pretrained != "": #Training from checkpoint(continue_roberta)
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_pretrained), strict=False)
        print('Loaded from checkpoint')
        dev_score, dev_output = ATLOP_evaluate(ATLOP_args, model, dev_features, tag="dev")
        print(dev_output)
        ATLOP_train(ATLOP_args, model, train_features, dev_features, test_features)
    elif args.load_path == "":  # Training
        ATLOP_train(ATLOP_args, model, train_features, dev_features, test_features)
    else:  # Testing
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = ATLOP_evaluate(ATLOP_args, model, dev_features, tag="dev")
        print(dev_output)
        pred = ATLOP_report(ATLOP_args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
