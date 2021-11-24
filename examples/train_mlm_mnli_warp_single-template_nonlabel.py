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

import os
import copy
import json
import sys
sys.path.append("../")
from utils import set_seed, random_mask_input_ids
from models.modeling_roberta import WARPPromptedRobertaForMaskedLM, WARPPromptedRobertaForSequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--n_prompts", default=20, type=int)
parser.add_argument("--backbone_model", default="roberta-large", type=str)
parser.add_argument("--pretrained_prompts_path", type = str)
parser.add_argument("--save_prompts_path", type = str)
parser.add_argument("--random_mask_rate", type = float, default = .15)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--cycle", default=50000, type=int)
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
parser.add_argument("--tensorboard_dir", type=str, default="./runs_mnli_mlm")
parser.add_argument("--tmult", default=1, type=int)
parser.add_argument("--comment", default='', type=str)
args = parser.parse_args()

# set seed
set_seed(args.seed)

# load data
dataset = load_dataset('multi_nli', split='train')
dataset = pd.DataFrame.from_dict({
    "premise": [d['premise'] for d in dataset],
    "hypothesis": [d['hypothesis'] for d in dataset],
    "gold_label": [('entailment', 'neutral', 'contradiction')[d['label']] for d in dataset]
})
valid_dataset = load_dataset('multi_nli', split='validation_matched')
valid_dataset = pd.DataFrame.from_dict({
    "premise": [d['premise'] for d in valid_dataset],
    "hypothesis": [d['hypothesis'] for d in valid_dataset],
    "gold_label": [('entailment', 'neutral', 'contradiction')[d['label']] for d in valid_dataset]
})

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.backbone_model)
token_ids_exception_list = [0, 1, 2, 50264]

# set device
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(f"device: {device}")

# initialize model
model = WARPPromptedRobertaForMaskedLM(
                                         pretrained_backbone_path = args.backbone_model,                 
                                         n_prompts = args.n_prompts, 
                                         seed_token_id_for_prompts_embeddings = 50264,
                                         pretrained_prompts_path = args.pretrained_prompts_path
                                        )

# move model to device
model.to(device)
# model.soft_prompts.prompts = model.soft_prompts.prompts.to(device)

# set up scheduler
optim = torch.optim.Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.learning_rate)
warmup = int(0.06 * args.cycle)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, args.cycle, T_mult=args.tmult)

def validate():
    model.eval()
    total_seen = 0
    total_loss = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(range(0, valid_dataset.shape[0], args.eval_batch_size)):
            exs = list(range(_, _+args.eval_batch_size))
            if exs[-1] >= valid_dataset.shape[0] or len(exs)<2:
                print('BREAKING!')
                break
            total_seen += len(exs)
            #valid_tokenized_ip = tokenizer(list(zip(valid_dataset.premise.tolist()[exs], valid_dataset.hypothesis.tolist()[exs])), return_tensors='pt', truncation=True, padding=True).to(device)

            valid_features = tokenizer([valid_dataset.premise.iloc[i] + "?" + valid_dataset.gold_label.iloc[i] + valid_dataset.hypothesis.iloc[i] for i in exs], return_tensors='pt', truncation=True, padding=True)
            valid_features["input_ids"] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 0), valid_features['input_ids']], 1)
            valid_features['labels'] = valid_features["input_ids"].detach().clone()
            # mask inputs
            valid_features['input_ids'] = random_mask_input_ids(valid_features['input_ids'], 50264, token_ids_exception_list)

            valid_features["input_ids"] = valid_features["input_ids"].to(device)
            valid_features["labels"] = valid_features["labels"].to(device)

            valid_features['attention_mask'] = torch.cat([torch.full((valid_features["input_ids"].shape[0], args.n_prompts), 1), valid_features['attention_mask']], 1)
            valid_features['attention_mask'] = valid_features['attention_mask'].to(device)

            #valid_final_op = torch.Tensor(valid_dataset.label.tolist()).type(torch.int64).cuda()
            outputs = model(**valid_features)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / total_seen
    print({"avg_loss": avg_loss})
    return avg_loss

# set up tensorboard
if args.use_tensorboard:
    writer = SummaryWriter(log_dir = f"{args.tensorboard_dir}/{args.comment}")

# training loop
model.eval()
optim.zero_grad()

for step in tqdm.tqdm(range(250000)):
    # exs = random.sample(range(0, dataset.shape[0]), bsz)
    exs = random.sample(range(0, dataset.shape[0]), args.train_batch_size)
    #ip = tokenizer([dataset.sentence.tolist()[_] for _ in exs], return_tensors='pt', truncation=True, padding=True)
    #op = [dataset.label.tolist()[_] for _ in exs]
    train_features = tokenizer([dataset.premise.iloc[i] + "?" + " <mask> " +  dataset.hypothesis.iloc[i] for i in exs], return_tensors='pt', truncation=True, padding=True)
    train_features["input_ids"] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 0), train_features['input_ids']], 1)
    train_features['labels'] = train_features["input_ids"].detach().clone()
    # mask inputs
    train_features['input_ids'] = random_mask_input_ids(train_features['input_ids'], 50264, token_ids_exception_list, args.random_mask_rate)

    train_features["input_ids"] = train_features["input_ids"].to(device)
    train_features["labels"] = train_features["labels"].to(device)

    train_features['attention_mask'] = torch.cat([torch.full((train_features["input_ids"].shape[0], args.n_prompts), 1), train_features['attention_mask']], 1)
    train_features['attention_mask'] = train_features['attention_mask'].to(device)

    outputs = model(**train_features)

    loss = outputs.loss / args.gradient_accumulation_steps
    loss.backward()
    if (step % args.gradient_accumulation_steps) == 0:
        optim.step()
        scheduler.step()
        optim.zero_grad()

    if step % 1000 == 0 or step == 249999:
        val_loss = validate()
        path_to_save_prompts = os.path.join(args.save_prompts_path, f"step_{step}")
        model.backbone.roberta.embeddings.word_embeddings.save_pretrained_soft_prompts(path_to_save_prompts)

        if args.use_tensorboard:
            writer.add_scalar('avg_loss_train', loss.item()/args.train_batch_size, step)
            writer.add_scalar('avg_loss_val', val_loss, step)
        
        # if step > 0 and step % 100000 == 0:
        #     raise Exception("100k steps has reached!!")
