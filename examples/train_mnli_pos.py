"""
MNLI task using method from "The Power of Scale for Parameter-Efficient Prompt Tuning"
"""

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

from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

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
    "gold_label": [('yes', 'neutral', 'instead')[d['label']] for d in dataset]
})
valid_dataset = load_dataset("glue", "mnli", split="validation_matched")
valid_dataset = pd.DataFrame.from_dict({
    "premise": [d['premise'] for d in valid_dataset],
    "hypothesis": [d['hypothesis'] for d in valid_dataset],
    "gold_label": [('yes', 'neutral', 'instead')[d['label']] for d in valid_dataset],
    "label": valid_dataset['label']
})

label_dict = {'yes': 0, 'neutral': 1, 'instead': 2}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.backbone_model)

# global/constant variable
mask_id = tokenizer.convert_tokens_to_ids("<mask>")

# set device
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(f"device: {device}")

# initialize model same as MLM for WARP, the delta is, the inputs will not be randomly masked
model = WARPPromptedRobertaForMaskedLM(
                                         pretrained_backbone_path = args.backbone_model,                 
                                         n_prompts = args.n_prompts, 
                                         seed_token_id_for_prompts_embeddings = mask_id,
                                         pretrained_prompts_path = args.pretrained_prompts_path
                                        )


# move model to device
model.to(device)

# set up scheduler
optim = torch.optim.Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.learning_rate)
warmup = int(0.06 * args.cycle)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, args.cycle, T_mult=args.tmult)

middle_string = " <mask> "

def validate():
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in tqdm.tqdm(range(0, valid_dataset.shape[0], args.eval_batch_size)):
            exs = list(range(_, _+args.eval_batch_size))
            if exs[-1] >= valid_dataset.shape[0]:
                exs = list(range(_, valid_dataset.shape[0]))

            valid_features = tokenizer([valid_dataset.premise.iloc[i] + "?" + middle_string + valid_dataset.hypothesis.iloc[i] for i in exs], return_tensors='pt', truncation=True, padding=True)
            valid_features["input_ids"] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 0), valid_features['input_ids']], 1)
            valid_features["input_ids"] = valid_features["input_ids"].to(device)
            valid_features['attention_mask'] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 1), valid_features['attention_mask']], 1)
            valid_features['attention_mask'] = valid_features['attention_mask'].to(device)

            outputs = model(**valid_features)

            logits = outputs.logits[valid_features["input_ids"] == mask_id]
            pred = logits.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            
            pred = [tokenizer.decode([gen_id], skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for gen_id in pred]

            preds.extend(pred)

    label2idx = {"yes": 0, "neutral": 1, "instead": 2}
    preds_idx = []
    for p in preds:
        if p.lower().strip() in label2idx:
            preds_idx.append(label2idx[p.lower().strip()])
        else:
            # print(f"prediction: // {p} //, predicted label out of known vocabulary!")
            preds_idx.append(3)
    eval_labels = valid_dataset['label']
    acc = accuracy_score(eval_labels, preds_idx)
    print(f"acc on validation dataset: {acc}")
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
    train_features = tokenizer([dataset.premise.iloc[i] + "?" + middle_string +  dataset.hypothesis.iloc[i] for i in exs], return_tensors='pt', truncation=True, padding=True)
    
    labels = tokenizer([dataset.premise.iloc[i] + "?" + f" {dataset.gold_label.iloc[i]} " +  dataset.hypothesis.iloc[i] for i in exs], return_tensors='pt', truncation=True, padding=True).input_ids
    labels = torch.cat([torch.full((labels.shape[0], args.n_prompts), 0), labels], 1)

    train_features["input_ids"] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 0), train_features['input_ids']], 1)
    train_features["input_ids"] = train_features["input_ids"].to(device)
    train_features['attention_mask'] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 1), train_features['attention_mask']], 1)
    train_features['attention_mask'] = train_features['attention_mask'].to(device)
    train_features['labels'] = labels
    train_features['labels'] = train_features['labels'].to(device)

    try:
        outputs = model(**train_features)
    except:
        print(f"{exs}")
        print(f"{train_features['input_ids'].shape}, {train_features['attention_mask'].shape}, {train_features['labels'].shape}")


    loss = outputs.loss / args.gradient_accumulation_steps
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
        else:
            cnt += 1

        # log into tensorboard
        if args.use_tensorboard:
            writer.add_scalar('avg train loss', loss.item()/args.train_batch_size, step)
            writer.add_scalar('acc val', acc, step)

        if cnt >= args.patience:
            best_soft_embedding.save_pretrained_soft_prompts(args.save_prompts_path)
            raise Exception("running out of patience!")
