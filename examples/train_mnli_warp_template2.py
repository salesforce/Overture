from datasets import load_dataset

from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import tqdm
import logging
import math
import pandas as pd
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import copy
import json
import sys
sys.path.append("../")
from utils import set_seed
from models.modeling_roberta import WARPPromptedRobertaForMaskedLM, WARPPromptedRobertaForSequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--n_prompts", default=20, type=int)
parser.add_argument("--backbone_model", default="roberta-large", type=str)
parser.add_argument("--pretrained_prompts_path", type = str)
parser.add_argument("--pretrained_classifier_path", type = str)
parser.add_argument("--save_prompts_path", type = str)
parser.add_argument("--save_classifier_path", type = str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--cycle", default=50000, type=int)
parser.add_argument("--patience", default=25, type=int)
parser.add_argument("--learning_rate", default=0.003, type=float)
parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available"
                    )

parser.add_argument("--use_tensorboard", action='store_true')
parser.add_argument("--tensorboard_dir", type=str, default="./runs_mnli_cls")

parser.add_argument("--tmult", default=1, type=int)
parser.add_argument("--comment", default='', type=str)
args = parser.parse_args()

# set seed
set_seed(args.seed)

# load data
dataset = load_dataset('glue', 'mnli', split='train')
dataset = pd.DataFrame.from_dict({
    "premise": [d['premise'] for d in dataset],
    "hypothesis": [d['hypothesis'] for d in dataset],
    "gold_label": [('entailment', 'neutral', 'contradiction')[d['label']] for d in dataset]
})
valid_dataset = load_dataset("glue", "mnli", split="validation_matched")
valid_dataset = pd.DataFrame.from_dict({
    "premise": [d['premise'] for d in valid_dataset],
    "hypothesis": [d['hypothesis'] for d in valid_dataset],
    "gold_label": [('entailment', 'neutral', 'contradiction')[d['label']] for d in valid_dataset]
})

label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.backbone_model)

# set device
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(f"device: {device}")

# initialize model
model = WARPPromptedRobertaForSequenceClassification(
                                                     pretrained_backbone_path = args.backbone_model,                 
                                                     n_prompts = args.n_prompts, 
                                                     seed_token_id_for_prompts_embeddings = 50264,
                                                     mask_token_id = 50264,
                                                     token_ids_for_classification_head = [1342, 12516, 10800], #'ent', 'neutral', 'cont'
                                                     pretrained_prompts_path = args.pretrained_prompts_path,
                                                     pretrained_classifier_path = args.pretrained_classifier_path
                                                     )


# move model to device
model.to(device)

# set up scheduler
optim = torch.optim.Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.learning_rate)
warmup = int(0.06 * args.cycle)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, args.cycle, T_mult=args.tmult)

middle_string = " <mask>"

def validate():
    model.eval()
    total_seen = 0
    total_correct = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(range(0, valid_dataset.shape[0], args.eval_batch_size)):
            exs = list(range(_, _+args.eval_batch_size))
            if exs[-1] >= valid_dataset.shape[0] or len(exs)<2:
                print('BREAKING!')
                break
            total_seen += len(exs)

            valid_features = tokenizer(["does " + valid_dataset.premise.iloc[i] + " mean that " + valid_dataset.hypothesis.iloc[i] + "?" + middle_string for i in exs], return_tensors='pt', truncation=True, padding=True)
            valid_features["input_ids"] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 0), valid_features['input_ids']], 1)
            valid_features["input_ids"] = valid_features["input_ids"].to(device)
            valid_features['attention_mask'] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 1), valid_features['attention_mask']], 1)
            valid_features['attention_mask'] = valid_features['attention_mask'].to(device)

            lbl = torch.Tensor([label_dict[valid_dataset.gold_label.iloc[i]] for i in exs]).type(torch.int64).cuda()        
            bleh = model(**valid_features)
            total_correct += torch.sum(bleh.argmax(1)==lbl).item()
    acc = total_correct/total_seen*100.
    print({"val_acc": acc})
    return acc

# set up tensorboard
if args.use_tensorboard:
    writer = SummaryWriter(log_dir = f"{args.tensorboard_dir}/{args.comment}")

# training loop
model.eval()
optim.zero_grad()
best_model = None
best_acc = .0
cnt = 0

for step in tqdm.tqdm(range(250000)):
    exs = random.sample(range(0, dataset.shape[0]), args.train_batch_size)
    train_features = tokenizer(["does " + dataset.premise.iloc[i] + " mean that " + dataset.hypothesis.iloc[i] + "?" + middle_string  for i in exs], return_tensors='pt', truncation=True, padding=True)
    train_features["input_ids"] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 0), train_features['input_ids']], 1)
    train_features["input_ids"] = train_features["input_ids"].to(device)
    train_features['attention_mask'] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 1), train_features['attention_mask']], 1)
    train_features['attention_mask'] = train_features['attention_mask'].to(device)

    lbl = torch.Tensor([label_dict[dataset.gold_label.iloc[i]] for i in exs]).type(torch.int64).to(device)

    bleh = model(**train_features)

    loss = torch.nn.functional.cross_entropy(bleh, lbl) / args.gradient_accumulation_steps
    loss.backward()
    if (step % args.gradient_accumulation_steps) == 0:
        optim.step()
        scheduler.step()
        optim.zero_grad()

    if step % 1000 == 0:
        acc = validate()
        if acc > best_acc:
            best_acc = acc
            cnt = 0
            best_soft_embedding = copy.deepcopy(model.backbone.roberta.embeddings.word_embeddings)
            best_classification_head = copy.deepcopy(model.classification_head)
        else:
            cnt += 1

        # log into tensorboard
        if args.use_tensorboard:
            writer.add_scalar('avg train loss', loss.item()/args.train_batch_size, step)
            writer.add_scalar('acc val', acc, step)

        if cnt >= args.patience:
            best_soft_embedding.save_pretrained_soft_prompts(args.save_prompts_path)
            best_classification_head.save_pretrained_classifier(args.save_classifier_path)
            raise Exception("running out of patience!")
