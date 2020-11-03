# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# Maps model class name to a tuple of model class
MODEL_CLASSES = [
    'AutoModel',
    'AutoModelWithLMHead',
    'AutoModelForSequenceClassification',
    'AutoModelForQuestionAnswering'
]

# List of pretrained models: https://huggingface.co/transformers/pretrained_models.html
# Pretrained model name to a tuple of input names, opset_version, use_external_data_format, optimization model type
MODELS = {
    # BERT
    "bert-base-uncased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-large-uncased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-large-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-multilingual-uncased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-multilingual-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-chinese": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-german-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-large-uncased-whole-word-masking": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-large-cased-whole-word-masking": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (["input_ids", "attention_mask",
                                                               "token_type_ids"], 11, False, "bert"),
    "bert-large-cased-whole-word-masking-finetuned-squad": (["input_ids", "attention_mask",
                                                             "token_type_ids"], 11, False, "bert"),
    "bert-base-cased-finetuned-mrpc": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-german-dbmdz-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "bert-base-german-dbmdz-uncased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    # todo: more models to add
    # GPT
    "openai-gpt": (["input_ids"], 11, False, "gpt2"),  # no past state inputs
    # GPT-2
    "gpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs
    "gpt2-medium": (["input_ids"], 11, False, "gpt2"),
    "gpt2-large":
    (["input_ids"], 11, True,
     "gpt2"),  # Model>2GB. Need use_external_data_format=True to export it. No past state inputs for GPT models.
    "gpt2-xl": (["input_ids"], 11, True, "gpt2"),
    "distilgpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs
    # Transformer-XL
    #"transfo-xl-wt103": (["input_ids"], 11, False, "bert"),
    # XLNet
    #"xlnet-base-cased": (["input_ids"], 12, False, "bert"), # Models uses Einsum, which need opset version 12 and PyTorch 1.5.0 or above.
    #"xlnet-large-cased": (["input_ids"], 12, False, "bert"), # Models uses Einsum, which need opset version 12 and PyTorch 1.5.0 or above.
    # XLM
    "xlm-mlm-en-2048": (["input_ids"], 11, True, "bert"),
    "xlm-mlm-ende-1024": (["input_ids"], 11, False, "bert"),
    "xlm-mlm-enfr-1024": (["input_ids"], 11, False, "bert"),
    "xlm-mlm-enro-1024": (["input_ids"], 11, False, "bert"),
    "xlm-mlm-xnli15-1024": (["input_ids"], 11, False, "bert"),
    "xlm-mlm-tlm-xnli15-1024": (["input_ids"], 11, False, "bert"),
    "xlm-clm-enfr-1024": (["input_ids"], 11, False, "bert"),
    "xlm-clm-ende-1024": (["input_ids"], 11, False, "bert"),
    "xlm-mlm-17-1280": (["input_ids"], 11, True, "bert"),
    "xlm-mlm-100-1280": (["input_ids"], 11, True, "bert"),
    # RoBERTa
    "roberta-base": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-large": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-large-mnli": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilroberta-base": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-base-openai-detector": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-large-openai-detector": (["input_ids", "attention_mask"], 11, False, "bert"),
    # DistilBERT
    "distilbert-base-uncased": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilbert-base-uncased-distilled-squad": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilbert-base-cased": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilbert-base-cased-distilled-squad": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilbert-base-german-cased": (["input_ids", "attention_mask"], 11, False, "bert"),
    "distilbert-base-multilingual-cased": (["input_ids", "attention_mask"], 11, False, "bert"),
    # CTRL
    "ctrl": (["input_ids"], 11, True, "bert"),
    # CamemBERT
    "camembert-base": (["input_ids"], 11, False, "bert"),
    # ALBERT
    # Models uses Einsum, which need opset version 12 and PyTorch 1.5.0 or above.
    "albert-base-v1": (["input_ids"], 12, False, "bert"),
    "albert-large-v1": (["input_ids"], 12, False, "bert"),
    "albert-xlarge-v1": (["input_ids"], 12, True, "bert"),
    "albert-xxlarge-v1": (["input_ids"], 12, True, "bert"),
    "albert-base-v2": (["input_ids"], 12, False, "bert"),
    "albert-large-v2": (["input_ids"], 12, False, "bert"),
    "albert-xlarge-v2": (["input_ids"], 12, True, "bert"),
    "albert-xxlarge-v2": (["input_ids"], 12, True, "bert"),
    # T5
    "t5-small": (["input_ids"], 12, False, "bert"),
    "t5-base": (["input_ids"], 12, False, "bert"),
    "t5-large": (["input_ids"], 12, True, "bert"),
    "t5-3b": (["input_ids"], 12, True, "bert"),
    "t5-11b": (["input_ids"], 12, True, "bert"),
    # XLM-RoBERTa
    "xlm-roberta-base": (["input_ids"], 11, False, "bert"),
    "xlm-roberta-large": (["input_ids"], 11, True, "bert"),
    # FlauBERT
    "flaubert/flaubert_small_cased": (["input_ids"], 11, False, "bert"),
    "flaubert/flaubert_base_uncased": (["input_ids"], 11, False, "bert"),
    "flaubert/flaubert_base_cased": (["input_ids"], 11, False, "bert"),
    "flaubert/flaubert_large_cased": (["input_ids"], 11, False, "bert"),
    # Bart
    "facebook/bart-large": (["input_ids"], 11, False, "bert"),
    "facebook/bart-base": (["input_ids"], 11, False, "bert"),
    "facebook/bart-large-mnli": (["input_ids"], 11, False, "bert"),
    "facebook/bart-large-cnn": (["input_ids"], 11, False, "bert"),
    #"facebook/mbart-large-en-ro": (["input_ids"], 11, True, "bert"),
    # DialoGPT
    "microsoft/DialoGPT-small": (["input_ids"], 11, False, "gpt2"),
    "microsoft/DialoGPT-medium": (["input_ids"], 11, False, "gpt2"),
    "microsoft/DialoGPT-large": (["input_ids"], 11, True, "gpt2"),
    # Reformer
    #"google/reformer-enwik8": (["input_ids"], 11, False, "bert"),
    #"google/reformer-crime-and-punishment": (["input_ids"], 11, False, "bert"),
    # MarianMT
    #"Helsinki-NLP/opus-mt-ROMANCE-en": (["input_ids"], 12, False, "bert"),
    # Longformer
    #"allenai/longformer-base-4096": (["input_ids"], 12, False, "bert"),
    #"allenai/longformer-large-4096": (["input_ids"], 12, False, "bert"),
}
