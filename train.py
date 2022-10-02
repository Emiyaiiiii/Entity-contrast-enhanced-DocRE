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
from model import DocREModel, SimCSEModel
from utils import set_seed, ATLOP_collate_fn, SimCSE_collecte_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb

# some bascal arguments
@dataclass
class CommonArguments():
    use_simcse_model: Optional[str] = field(
        default='no',
        metadata={
            "choices": ['yes', 'no'],
            "help": "if use SimCSE model to fine-tune bert model"
        }
    )
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

# arguments of SimCSE
@dataclass
class SimCSEArguments():
    simcse_save_path: Optional[str] = field(
        default='/data/wh/workspace/ATLOP/models/simcse_sup_bert.pt'
    )
    simcse_load_path: Optional[str] = field(
        default='/data/wh/workspace/ATLOP/models/simcse_sup_bert.pt'
    )
    simcse_epoch: Optional[int] = field(
        default=5
    )
    simcse_batch_size: Optional[int] = field(
        default=2
    )
    simcse_learning_rate: Optional[float] = field(
        default=1e-5
    )
    simcse_pooling: Optional[str] = field(
        default='cls',
        metadata={
            "choices": ['cls', 'first-last-avg', 'last-avg']
        }
    )

def SimCSE_train(args, model, train_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        train_dataloader = DataLoader(features, batch_size=args.simcse_batch_size, shuffle=True, collate_fn=SimCSE_collecte_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        global best
        # early_stop_batch = 0
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader, start=1):
                model.train()
                # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
                real_batch_num = batch[0].shape[0]
                input_ids = batch[0].view(real_batch_num * 3, -1).to(args.device)
                attention_mask = batch[1].view(real_batch_num * 3, -1).to(args.device)
                # 训练
                inputs = {'input_ids': input_ids,
                          'attention_mask': attention_mask
                        }
                loss = model(**inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_steps += 1
                # 评估
                if optimizer % 100 == 0:
                    print("SimCSE_loss:" + str(loss.item()))
    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.simcse_learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.simcse_epoch, num_steps)

def SimCSE_evaluate(args, model, features):
    model.eval()
    dataloader = DataLoader(features, batch_size=args.simcse_batch_size, shuffle=False, collate_fn=SimCSE_collecte_fn, drop_last=False)
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(args.device)
            source_attention_mask = source['attention_mask'].squeeze(1).to(args.device)
            source_pred = model(source_input_ids, source_attention_mask)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(args.device)
            target_attention_mask = target['attention_mask'].squeeze(1).to(args.device)
            target_pred = model(target_input_ids, target_attention_mask)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
            # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation

# arguments of ATLOP
@dataclass
class ATLOPArguments():
    save_path: Optional[str] = field(
        default=""
    )
    load_path: Optional[str] = field(
        default=""
    )
    load_pretrained: Optional[str] = field(
        default=""
    )
    axial_attention: Optional[str] = field(
        default="multi_external",
        metadata={
            "choices": ['none', 'self_attention', 'single_external', 'multi_external'],
            "help": "type of AxialAttention"
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
                # input_ids:Bert输入的原句
                # attention_mask:Bert attention
                # labels：训练最终的结果，即真正的句间关系信息
                # entity_pos：实体对应所有提及所在句子、位置、内容以及类型等信息
                # hts:一个关系对里头实体尾实体对应下标
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
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

    # #if len(ans) > 0:
    # if tag=='dev':
    #     best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(ans, args.data_dir, args.train_file, args.dev_file)
    # elif tag=='train':
    #     best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(ans, args.data_dir, args.train_file, args.train_file)
    # output = {
    #     tag + "_F1": best_f1 * 100,
    #     tag + "_F1_ign": best_f1_ign * 100,
    #     tag + "_P": best_p * 100,
    #     tag + "_R": best_r * 100,
    # }
    # return best_f1, output

    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    else:
        best_f1, _, best_f1_ign, _ =0, 0, 0, 0
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
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
    parser = HfArgumentParser((CommonArguments, SimCSEArguments, ATLOPArguments))
    args, SimCSE_args, ATLOP_args= parser.parse_args_into_dataclasses()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device, SimCSE_args.device, ATLOP_args.device = device, device, device
    args.n_gpu, SimCSE_args.n_gpu, ATLOP_args.n_gpu = torch.cuda.device_count(), torch.cuda.device_count(), torch.cuda.device_count()
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
    if os.path.exists(os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix)) and os.path.exists(os.path.join(args.data_dir, args.train_file + "train_simcse_features" + suffix)):
        train_features = torch.load(os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix))
        train_simcse_features = torch.load(os.path.join(args.data_dir, args.train_file + "train_simcse_features" + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features , train_simcse_features= read(train_file, tokenizer, max_seq_length=args.max_seq_length, use_simcse_model=args.use_simcse_model, axial_attention=ATLOPArguments.axial_attention)
        torch.save(train_features, os.path.join(args.data_dir, args.train_file + ATLOP_args.axial_attention + suffix))
        torch.save(train_simcse_features, os.path.join(args.data_dir, args.train_file + "train_simcse_features" + suffix))
        print('Created and saved new train features')
    # dev_features   
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix)) and os.path.exists(os.path.join(args.data_dir, args.dev_file + "dev_simcse_features" + suffix)):
        dev_features= torch.load(os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix))
        dev_simcse_features = torch.load(os.path.join(args.data_dir, args.dev_file + "dev_simcse_features" + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features ,dev_simcse_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, use_simcse_model=args.use_simcse_model, axial_attention=ATLOPArguments.axial_attention)
        torch.save(dev_features, os.path.join(args.data_dir, args.dev_file + ATLOP_args.axial_attention + suffix))
        torch.save(dev_simcse_features, os.path.join(args.data_dir, args.dev_file + "dev_simcse_features" + suffix))
        print('Created and saved new dev features')
    # test_features
    if os.path.exists(os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix)):
        test_features = torch.load(os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features, test_simcse_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, use_simcse_model=args.use_simcse_model, axial_attention=ATLOPArguments.axial_attention)
        torch.save(test_features, os.path.join(args.data_dir, args.test_file + ATLOP_args.axial_attention + suffix))
        print('Created and saved new test features')

    #SimCSE re-training Bert and save
    if args.use_simcse_model == "yes":
        if SimCSE_args.simcse_load_path == "":
            # load model
            assert SimCSE_args.simcse_pooling in ['cls', 'last-avg', 'first-last-avg']
            simcseModel = SimCSEModel(config, pretrained_model=model, pooling=SimCSE_args.simcse_pooling)
            simcseModel.to(args.device)
            # Train
            SimCSE_train(SimCSE_args, simcseModel, train_simcse_features, dev_simcse_features)
            torch.save(model.state_dict(), SimCSE_args.simcse_save_path)
        else:
            model.load_state_dict(torch.load(SimCSE_args.simcse_load_path))
            print("load simcse model")

    #Training
    wandb.init(project="ATLOP", entity="15346186000", config=ATLOP_args)
    
    set_seed(ATLOP_args)
    model = DocREModel(config, model, num_labels=ATLOP_args.num_labels, axial_attention=ATLOP_args.axial_attention)
    model.to(device)

    if ATLOP_args.load_pretrained != "": #Training from checkpoint(continue_roberta)
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(ATLOP_args.load_pretrained), strict=False)
        print('Loaded from checkpoint')
        dev_score, dev_output = ATLOP_evaluate(ATLOP_args, model, dev_features, tag="dev")
        print(dev_output)
        ATLOP_train(ATLOP_args, model, train_features, dev_features, test_features)
    elif ATLOP_args.load_path == "":  # Training
        ATLOP_train(ATLOP_args, model, train_features, dev_features, test_features)
    else:  # Testing
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(ATLOP_args.load_path))
        dev_score, dev_output = ATLOP_evaluate(ATLOP_args, model, dev_features, tag="dev")
        print(dev_output)
        pred = ATLOP_report(ATLOP_args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
