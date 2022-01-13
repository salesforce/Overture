import os
import math
import logging
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

import sys

sys.path.append("../")
from soft_prompts import PromptedWordEmbeddings

logger = logging.getLogger(__name__)


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


# 1, can be used for MLM
# 2, can also be used for MLM-style classification without additional classification layer, like "The Power of Scale for Parameter-Efficient Prompt Tuning" paper
class WARPPromptedRobertaForMaskedLM(nn.Module):
    def __init__(
        self,
        pretrained_backbone_path,
        n_prompts,
        seed_token_id_for_prompts_embeddings,
        pretrained_prompts_path=None,
    ):
        """
        pretrained_backbone_path: str, path to or name of backbone model, e.g. roberta-large;
        n_prompts: int, number of prompts;
        seed_token_id_for_prompts_embeddings: int, use embedding of a specific token to initialize prompts weights, usually use mask token.
        """
        super(WARPPromptedRobertaForMaskedLM, self).__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_backbone_path)
        self.n_prompts = n_prompts
        # freeze backbone model
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        original_word_embeddings = self.backbone.roberta.embeddings.word_embeddings
        prompted_word_embeddings = PromptedWordEmbeddings(
            original_word_embeddings,
            n_prompts,
            hidden_size,
            seed_token_id_for_prompts_embeddings,
        )
        if pretrained_prompts_path is not None:
            prompted_word_embeddings.load_from_pretrained_soft_prompts(
                pretrained_prompts_path
            )
            logger.info(
                f"loaded pretrained soft prompts from: {pretrained_prompts_path}"
            )

        self.backbone.roberta.embeddings.word_embeddings = prompted_word_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        return self.backbone(input_ids, attention_mask, labels=labels)


# classification head modified from https://github.com/huggingface/transformers/blob/5e3b4a70d3d17f2482d50aea230f7ed42b3a8fd0/src/transformers/models/roberta/modeling_roberta.py#L1123
class RobertaClassificationHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, ori_lm_head, weight_tensors, hidden_size, prediction_dim):
        """
        ori_lm_head: original lm_head from roberta model, can be accessed by model.lm_head;
        weight_tensors: initialize final classifier layer with the specified weight tensors, usually from verbalzier token embeddings;
        hidden_size, int, backbone model hidden size;
        prediction_dim, int, output dimension of classifier layer.
        """
        super().__init__()
        self.dense = ori_lm_head.dense
        self.layer_norm = ori_lm_head.layer_norm
        self.bias = ori_lm_head.bias

        self.classifier = torch.nn.Linear(hidden_size, prediction_dim, bias=True)
        self.classifier.weight = weight_tensors

    def load_from_pretrained_classifier(self, pretrained_classifier_path):
        path = os.path.join(pretrained_classifier_path, "classifier.pt")
        pretrained_classifier = torch.load(path)
        if (
            pretrained_classifier.weight.shape == self.classifier.weight.shape
            and pretrained_classifier.bias.shape == self.classifier.bias.shape
        ):
            self.classifier = pretrained_classifier
            logger.info(
                f"loaded pretrained classifier from {pretrained_classifier_path}"
            )
        else:
            raise Exception(
                f"pretrained classifier weights dimension: {pretrained_classifier.weight.shape}, bias dimension: {pretrained_classifier.bias.shape} \
                                but classifier initialized with {self.classifier.weight.shape} and {self.classifier.bias.shape}"
            )

    def save_pretrained_classifier(self, save_directory):
        path = os.path.join(save_directory, "classifier.pt")
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)
        torch.save(self.classifier, path)
        logger.info(f"saved trained classifier at: {save_directory}")

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        return self.classifier(x)

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class WARPPromptedRobertaForSequenceClassification(nn.Module):
    def __init__(
        self,
        pretrained_backbone_path,
        n_prompts,
        seed_token_id_for_prompts_embeddings,
        mask_token_id,
        token_ids_for_classification_head,
        pretrained_prompts_path=None,
        pretrained_classifier_path=None,
    ):
        """
        pretrained_backbone_path: str, path to or name of backbone model, e.g. roberta-large;
        n_prompts: int, number of prompts;
        seed_token_id_for_prompts_embeddings: int, use embedding of a specific token to initialize prompts weights, usually use mask token;
        mask_token_id: int, token id for mask token, 50264 for huggingface roberta model;
        token_ids_for_classification_head: list of int, used for initilize classifier weights;
        pretrained_prompts_path: str or None, path to pretrained prompts;
        pretrained_classifier_path: str or None, path to pretrained classifier layer.
        """
        super(WARPPromptedRobertaForSequenceClassification, self).__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_backbone_path)
        self.n_prompts = n_prompts
        self.mask_token_id = mask_token_id
        # freeze backbone model
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False

        # modify embedding layer for soft prompts
        hidden_size = self.backbone.config.hidden_size
        original_word_embeddings = self.backbone.roberta.embeddings.word_embeddings
        prompted_word_embeddings = PromptedWordEmbeddings(
            original_word_embeddings,
            n_prompts,
            hidden_size,
            seed_token_id_for_prompts_embeddings,
        )
        if pretrained_prompts_path is not None:
            prompted_word_embeddings.load_from_pretrained_soft_prompts(
                pretrained_prompts_path
            )

        self.backbone.roberta.embeddings.word_embeddings = prompted_word_embeddings

        # classification head
        weights4lm_head = torch.nn.Parameter(
            self.backbone.roberta.embeddings.word_embeddings.ori_emb.weight[
                token_ids_for_classification_head
            ]
        )
        prediction_dim = len(token_ids_for_classification_head)
        self.classification_head = RobertaClassificationHead(
            self.backbone.lm_head, weights4lm_head, hidden_size, prediction_dim
        )
        if pretrained_classifier_path is not None:
            self.classification_head.load_from_pretrained_classifier(
                pretrained_classifier_path
            )

        # remove original lm_head
        del self.backbone.lm_head

    def forward(self, input_ids, attention_mask):
        before_classifier = self.backbone.roberta(input_ids, attention_mask)[0]
        mask_token_locations = torch.where(input_ids == self.mask_token_id)
        return self.classification_head(before_classifier[mask_token_locations])


class WARPPromptedRobertaForQuestionAnswering(nn.Module):
    def __init__(
        self,
        pretrained_backbone_path,
        n_prompts,
        seed_token_id_for_prompts_embeddings,
        pretrained_prompts_path=None,
        freeze_qa_outputs_layer=True,
    ):
        """
        pretrained_backbone_path: str, path to or name of backbone model, e.g. roberta-large;
        n_prompts: int, number of prompts;
        seed_token_id_for_prompts_embeddings: int, use embedding of a specific token to initialize prompts weights, usually use mask token.
        """
        super(WARPPromptedRobertaForQuestionAnswering, self).__init__()
        self.backbone = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_backbone_path
        )
        self.n_prompts = n_prompts
        # freeze backbone model and/except the final qa_output layer
        for n, p in self.backbone.named_parameters():
            p.requires_grad = False
            if "qa_outputs" in n and not freeze_qa_outputs_layer:
                p.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        original_word_embeddings = self.backbone.roberta.embeddings.word_embeddings
        prompted_word_embeddings = PromptedWordEmbeddings(
            original_word_embeddings,
            n_prompts,
            hidden_size,
            seed_token_id_for_prompts_embeddings,
        )
        if pretrained_prompts_path is not None:
            prompted_word_embeddings.load_from_pretrained_soft_prompts(
                pretrained_prompts_path
            )
            logger.info(
                f"loaded pretrained soft prompts from: {pretrained_prompts_path}"
            )

        self.backbone.roberta.embeddings.word_embeddings = prompted_word_embeddings

    def forward(
        self, input_ids, attention_mask, start_positions=None, end_positions=None
    ):
        return self.backbone(
            input_ids,
            attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
