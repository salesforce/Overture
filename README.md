# Project Overture - A Prompt-Tuning Library for Researchers
> Why name it **Overture**? An overture in music is an orchestral piece at the beginning which sets the mood and tone for what's about to come. We think of prompt tuning as analogous to that; in the types of prompt tuning methods we consider, a prompt is prepended to the input that sets the tone for the downstream task. 
<p>
    <img src="figures/overture_logo.png" width="220" height="240" />
</p>

# Introduction
Prompt Tuning has recently become an important research direction in Natural Language Processing. In contrast to classical fine-tuning, which involves optimizing the weights of the entire network, (one style of) prompt tuning keeps the large language model (a.k.a. the "backbone") frozen and instead prepends a few learnable vectors to each input which are learnt in order to accomplish a task. This brings the number of parameters to train from O(millions) down to a few thousand while still achieving similar levels of performance. There are other benefits that have been found in the research community for prompt tuned models when compared to classically trained models. 

# Methods Supported
The repository leverages the HuggingFace Transformers repository and currently, we support [WARP-like](https://arxiv.org/abs/2101.00121) prompt-tuning for masked language modeling(MLM), text classification models, and extractive question answering (e.g., SQuAD). We plan on adding support for [Seq2Seq prompt-tuning](https://arxiv.org/abs/2104.08691v1) soon. If there is any other algorithm/method that you would like for us to prioritize, please write to us or file a feature request. Finally, we refer an interested reader to the [excellent survey](http://pretrain.nlpedia.ai/) on the topic for the various types of prompt tuning methods and their history. 

# Some Potential Extensions
Here are some research ideas one could experiment with our codebase. Since the community is evolving rapidly, it is entirely possible that some of these ideas have already been studied. Please file an issue if that is the case, or if you want to contribute more ideas. 

1. Does prompt tuning on a multilingual backbone (e.g., mBERT or XLM) lead to models that can perform cross-lingual zero-shot transfer?
2. How can we make the prompts more interpretable? Could adding a loss to make the prompt vectors be close to existing word embeddings help?
3. Can prompts learned for BERT-Large help learn prompts for RoBERTa-Large? 

# Design Choices & Other Similar Libraries 

Fundamentally, we designed the repository for researchers to easily experiment with ideas within the realm of prompt-tuning. As such, we intentionally do not abstract away the sub-components. The repository is intended to be a fork-and-edit library and is designed to be easily extensible for the kinds of projects we anticipated people to use the library for. 

A recently released library, [OpenPrompt](https://github.com/thunlp/OpenPrompt), is also intended to be a library for prompt tuning and we refer an interested practitioner to their repository for further exploration and comparisons. OpenPrompt may be a better fit for those who seek greater abstraction.

# How to Use
Inside the examples folder, we provide training code for RoBERTa-Large model on the MNLI dataset (in the style of [WARP](https://arxiv.org/abs/2101.00121)). To start training: 
```bash 
CUDA_VISIBLE_DEVICES=0 python train_warp_mnli.py --save_prompts_path dir_to_save_prompts --save_classifier_path dir_to_save_classifier 
```

After training, user should expect the model performance (accuracy) to be 87-89, which matches the original [WARP](https://arxiv.org/abs/2101.00121) paper results.

### Dev environment
- Python 3.8.5
- transfomers 4.11.2
- A-100 GPU, CUDA Version: 11.0

### API
```python
from models.modeling_roberta import WARPPromptedRobertaForMaskedLM, WARPPromptedRobertaForSequenceClassification
from utils import random_mask_input_ids

# initialize model for MNLI task
model = WARPPromptedRobertaForSequenceClassification(
                                                     pretrained_backbone_path = "roberta-large",                 
                                                     n_prompts = 8, 
                                                     seed_token_id_for_prompts_embeddings = 50264, # token id for "<mask>"
                                                     mask_token_id = 50264,
                                                     token_ids_for_classification_head = [1342, 12516, 10800], # 'ent', 'neutral', 'cont'
                                                     pretrained_prompts_path = None,
                                                     pretrained_classifier_path = None
                                                     )
                                                     
# initialize model for masked language modeling (MLM)
model = WARPPromptedRobertaForMaskedLM(
                                         pretrained_backbone_path = "roberta-large",                 
                                         n_prompts = 8, 
                                         seed_token_id_for_prompts_embeddings = 50264,
                                         pretrained_prompts_path = None
                                        )
                                        
# prepad input ids before feeding into model
features = tokenizer([str_1, str_2, ..., str_n], return_tensors='pt', truncation=True, padding=True)
features["input_ids"] = torch.cat([torch.full((features["input_ids"].shape[0], n_prompts), 0), features['input_ids']], 1)

# randomly mask input ids for MLM task
features['input_ids'] = random_mask_input_ids(features['input_ids'], mask_token_id, prob = .15)
```

### Reference
- [WARP: Word-level Adversarial ReProgramming](https://aclanthology.org/2021.acl-long.381.pdf)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691v1)
