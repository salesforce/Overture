import torch
import random
import numpy as np


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


# generate randomly masked input_ids for MLM task
# modified from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
def random_mask_input_ids(input_ids, mask_token_id, exceptions, prob=0.15):
    """
    exceptions: list, token ids that should not be masked
    """
    probs = torch.rand(input_ids.shape)
    mask = probs < prob
    for ex_id in exceptions:
        mask = mask * (input_ids != ex_id)
    selection = []
    for i in range(input_ids.shape[0]):
        selection.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(input_ids.shape[0]):
        input_ids[i, selection[i]] = mask_token_id
    return input_ids
