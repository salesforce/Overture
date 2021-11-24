import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PromptedWordEmbeddings(nn.Module):
    def __init__(self, 
                original_word_embeddings, 
                n_prompts, 
                hidden_size, 
                seed_token_id_for_prompts_embeddings):
        """
        original_word_embeddings: word embedding layer from backbone transformer model;
        n_prompts: int, number of soft prompts;
        hidden_size: int, should be same as backbone transformer model hidden size for embedding;
        seed_token_id_for_prompts_embeddings: soft prompts will be initialized with the weights of this token embedding, usually use mask token.
        """
        super(PromptedWordEmbeddings, self).__init__()
        self.ori_emb = original_word_embeddings
        self.n_prompts = n_prompts
        self.soft_prompts = torch.zeros(n_prompts, hidden_size) + original_word_embeddings.weight[seed_token_id_for_prompts_embeddings].clone().detach()
        self.soft_prompts = nn.Parameter(self.soft_prompts, requires_grad = True)
        logger.info(f"initialized soft prompts with dimension: {self.soft_prompts.shape}")

    def load_from_pretrained_soft_prompts(self, pretrained_prompts_path):
        pretrained_soft_prompts = torch.load(f"{pretrained_prompts_path}/prompts.pt")
        if pretrained_soft_prompts.shape[0] == self.n_prompts:
            self.soft_prompts = pretrained_soft_prompts
            logger.info(f"loaded pretrained soft prompts from {pretrained_prompts_path}")
        else:
            raise Exception(f"pretrained soft prompts dimension: {pretrained_soft_prompts.shape}, but initialized with {self.soft_prompts.shape}")
        
    def save_pretrained_soft_prompts(self, save_directory):
        path = os.path.join(save_directory, 'prompts.pt')
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)
        torch.save(self.soft_prompts, path)
        logger.info(f"saved trained soft prompts at {save_directory}")
        
    def forward(self, prepadded_input_ids):
        """
        prepadded_input_ids: input_ids after tokenization + prepadded tensors as placeholder for prompts
                             e.g. torch.cat([torch.full((features["input_ids"].shape[0], n_prompts), 0), features['input_ids']], 1) 
        """
        emb = self.ori_emb(prepadded_input_ids[:, self.n_prompts:])
        expanded_prompts = self.soft_prompts.repeat(prepadded_input_ids.shape[0], 1, 1)
        return torch.cat([expanded_prompts, emb], 1)