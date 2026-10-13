# Fine Tune LLMs

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

- What is a Transformer model?
- What is DistilBERT?
- What is text classification?
- What is sentiment analysis?
- What is Fine-Tuning?
- Why is fine-tuning needed?
- Why is tokenization required?
- What is subword tokenization?
- What is the role of attention masks?
- Why is truncation used?
- What is dynamic padding?
- Why do we compute weighted metrics like F1, precision, and recall?
- What does a confusion matrix show?
- What is a Hugging Face inference pipeline used for?
- How can we push a fine-tuned model to the Hugging Face Hub?

## 0-load_dataset.py

### 0. Load Dataset for Sentiment Analysis

Write a function `load_emotion_dataset()` that loads the Emotion dataset using the Hugging Face `datasets` library.

Returns:
- `dataset` (`DatasetDict`): The full dataset object containing the train, validation, and test splits.

Dataset: [https://huggingface.co/datasets/dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)

### Main 1

```bash
$ cat 0-main_1.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset

dataset = load_emotion_dataset()

print(type(dataset))

print("\nDataset Structure:")
print(dataset)

print("\nDataset Column Names:")
print(dataset.column_names)

print("\nDataset Shape:")
print(dataset.shape)

$ ./0-main_1.py
<class 'datasets.dataset_dict.DatasetDict'>

Dataset Structure:
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 16000
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
})

Dataset Column Names:
{'train': ['text', 'label'], 'validation': ['text', 'label'], 'test': ['text', 'label']}

Dataset Shape:
{'train': (16000, 2), 'validation': (2000, 2), 'test': (2000, 2)}
```

### Main 2

```bash
$ cat 0-main_2.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
import numpy as np

dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

for split_name in dataset.keys():
    print(f"\n{split_name.capitalize()} Set Label Distribution:")
    split_labels = np.array(dataset[split_name]['label'])
    for idx, name in enumerate(label_names):
        count = np.sum(split_labels == idx)
        percentage = 100 * count / len(split_labels)
        print(f"  {name:<10}: {count} samples ({percentage:.2f}%)")

    print(f"\nSample Examples from {split_name.capitalize()} Set:")
    for i in range(min(5, len(dataset[split_name]))):
        text = dataset[split_name]['text'][i]
        label_id = dataset[split_name]['label'][i]
        emotion = label_names[label_id]
        print(f"{i+1}. [{emotion.upper()}] {text}")

$ ./0-main_2.py

Train Set Label Distribution:
  sadness   : 4666 samples (29.16%)
  joy       : 5362 samples (33.51%)
  love      : 1304 samples (8.15%)
  anger     : 2159 samples (13.49%)
  fear      : 1937 samples (12.11%)
  surprise  : 572 samples (3.58%)

Sample Examples from Train Set:
1. [SADNESS] i didnt feel humiliated
2. [SADNESS] i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake
3. [ANGER] im grabbing a minute to post i feel greedy wrong
4. [LOVE] i am ever feeling nostalgic about the fireplace i will know that it is still on the property
5. [ANGER] i am feeling grouchy

Validation Set Label Distribution:
  sadness   : 550 samples (27.50%)
  joy       : 704 samples (35.20%)
  love      : 178 samples (8.90%)
  anger     : 275 samples (13.75%)
  fear      : 212 samples (10.60%)
  surprise  : 81 samples (4.05%)

Sample Examples from Validation Set:
1. [SADNESS] im feeling quite sad and sorry for myself but ill snap out of it soon
2. [SADNESS] i feel like i am still looking at a blank canvas blank pieces of paper
3. [LOVE] i feel like a faithful servant
4. [ANGER] i am just feeling cranky and blue
5. [JOY] i can have for a treat or if i am feeling festive

Test Set Label Distribution:
  sadness   : 581 samples (29.05%)
  joy       : 695 samples (34.75%)
  love      : 159 samples (7.95%)
  anger     : 275 samples (13.75%)
  fear      : 224 samples (11.20%)
  surprise  : 66 samples (3.30%)

Sample Examples from Test Set:
1. [SADNESS] im feeling rather rotten so im not very ambitious right now
2. [SADNESS] im updating my blog because i feel shitty
3. [SADNESS] i never make her separate from me because i don t ever want her to feel like i m ashamed with her
4. [JOY] i left with my bouquet of red and yellow tulips under my arm feeling slightly more optimistic than when i arrived
5. [SADNESS] i was feeling a little vain when i did this one
```

## 1-load_distilbert.py

### 1. Load Pretrained DistilBERT for Text Classification

Write a function `load_distilbert(model_name, num_classes, id2label, label2id)` that loads a tokenizer and a sequence classification model for text classification.

Arguments:
- `model_name` (`str`): Name of the pre-trained DistilBERT to load.
- `num_classes` (`int`): Total number of output classes for the classification task.
- `id2label` (`dict`): Dictionary mapping numeric label IDs to human-readable label names.
- `label2id` (`dict`): Dictionary mapping label names to their corresponding numeric IDs.

Returns:
- `tokenizer`: The tokenizer for preprocessing text inputs.
- `model`: The sequence classification model configured with the correct number of classes and label mappings.

### Main 1: View Tokenizer details

```bash
$ cat 1-main_1.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

print("Tokenizer type:", type(tokenizer))
print("\nTokenizer Vocabulary Size:", tokenizer.vocab_size)
print("Tokenizer Max Input Length:", tokenizer.model_max_length)
print("Tokenizer Model Input Names:", tokenizer.model_input_names)

$ ./1-main_1.py
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Tokenizer type: <class 'transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast'>

Tokenizer Vocabulary Size: 30522
Tokenizer Max Input Length: 512
Tokenizer Model Input Names: ['input_ids', 'attention_mask']
```

### Main 2: View Model details

```bash
$ cat 1-main_2.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

_, model = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

print("Model type:", type(model))
print("\nModel Configuration:")
print(model.config)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

$ ./1-main_2.py
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Model type: <class 'transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification'>

Model Configuration:
DistilBertConfig {
  "activation": "gelu",
  "architectures": [
    "DistilBertForMaskedLM"
  ],
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "dtype": "float32",
  "hidden_dim": 3072,
  "id2label": {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
  },
  "initializer_range": 0.02,
  "label2id": {
    "anger": 3,
    "fear": 4,
    "joy": 1,
    "love": 2,
    "sadness": 0,
    "surprise": 5
  },
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "tie_weights_": true,
  "transformers_version": "4.56.1",
  "vocab_size": 30522
}


Total parameters: 66,958,086
Trainable parameters: 66,958,086
```

## 2-tokenize_and_map.py

### 2. Tokenize Dataset

Write a function `tokenize_and_map(dataset, tokenizer, max_length, truncation, batched)` that tokenizes the text inputs of the entire Emotion dataset using a pretrained tokenizer.

The function should truncate sequences so that they do not exceed a specified maximum length.

The function should process multiple examples at once during the mapping process to speed up tokenization.

Arguments:
- `dataset` (`DatasetDict`): The dataset with `'train'`, `'validation'`, and `'test'` splits.
- `tokenizer`: Pretrained tokenizer.
- `max_length` (`int`): Maximum token length for truncation.
- `truncation` (`bool`): Whether to truncate sequences longer than `max_length`.
- `batched` (`bool`): Whether to process the dataset in batches.

Returns:
- `tokenized_train`, `tokenized_val`, `tokenized_test`: The tokenized train, validation, and test splits, ready to be used as model input.

### Test Example

```bash
$ cat 2-main.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map

from transformers import logging
logging.set_verbosity_error()

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=13,
    truncation=True,
    batched=True
)

# Inspect first examples
print("First tokenized train example:", tokenized_train[0])
print("First tokenized validation example:", tokenized_val[0])
print("First tokenized test example:", tokenized_test[0])

$ ./2-main.py
First tokenized train example: {'text': 'i didnt feel humiliated', 'label': 0, 'input_ids': [101, 1045, 2134, 2102, 2514, 26608, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
First tokenized validation example: {'text': 'im feeling quite sad and sorry for myself but ill snap out of it soon', 'label': 0, 'input_ids': [101, 10047, 3110, 3243, 6517, 1998, 3374, 2005, 2870, 2021, 5665, 10245, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
First tokenized test example: {'text': 'im feeling rather rotten so im not very ambitious right now', 'label': 0, 'input_ids': [101, 10047, 3110, 2738, 11083, 2061, 10047, 2025, 2200, 12479, 2157, 2085, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## 3-dynamic_padding.py

### 3. Dynamic Padding

Write a function `create_data_collator(tokenizer)` that sets up a dynamic padding mechanism for tokenized inputs using a pretrained tokenizer.

The function should:
- Automatically pad sequences in each batch to match the length of the longest sequence.
- Reduce unnecessary padding to improve training efficiency and memory usage.

Arguments:
- `tokenizer`: Pretrained tokenizer (e.g., DistilBERT tokenizer)

Returns:
- `data_collator`: `DataCollatorWithPadding` instance that dynamically pads inputs per batch.

### Test Example

```bash
$ cat 3-main.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map
create_data_collator = __import__('3-dynamic_padding').create_data_collator

from transformers import logging
logging.set_verbosity_error()

dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {name: idx for idx, name in enumerate(label_names)}
id_to_label = {idx: name for idx, name in enumerate(label_names)}
num_classes = len(label_names)
model_name = "distilbert-base-uncased"

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=128,
    truncation=True,
    batched=True
)

data_collator = create_data_collator(tokenizer)

def get_batch(start_idx, batch_size=4):
  return [
      {k: tokenized_train[i][k] for k in ["input_ids", "attention_mask"]}
      for i in range(start_idx, start_idx + batch_size)
  ]

batch1_examples = get_batch(0)
batch2_examples = get_batch(4)

batch1 = data_collator(batch1_examples)
batch2 = data_collator(batch2_examples)


for i, batch in enumerate([batch1, batch2], 1):
  print(f"Batch {i} shapes:")
  print("  input_ids:", batch['input_ids'].shape)
  print("  Input IDs:")
  print(batch['input_ids'])
  print("  attention_mask:", batch['attention_mask'].shape)
  print()

  ## Decode token IDs back to text ##
  print(f"  Decoded texts from Batch {i}:")
  for j, ids in enumerate(batch['input_ids']):
    decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"    Example {j+1}: {decoded_text}")
  print()

  ## Show with special tokens ##
  print(f"  Decoded texts (with special tokens):")
  for j, ids in enumerate(batch['input_ids']):
    decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"    Example {j+1}: {decoded_text}")
  print()

$ ./3-main.py
Batch 1 shapes:
  input_ids: torch.Size([4, 23])
  Input IDs:
tensor([[  101,  1045,  2134,  2102,  2514, 26608,   102,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [  101,  1045,  2064,  2175,  2013,  3110,  2061, 20625,  2000,  2061,
          9636, 17772,  2074,  2013,  2108,  2105,  2619,  2040, 14977,  1998,
          2003,  8300,   102],
        [  101, 10047,  9775,  1037,  3371,  2000,  2695,  1045,  2514, 20505,
          3308,   102,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [  101,  1045,  2572,  2412,  3110, 16839,  9080, 12863,  2055,  1996,
         13788,  1045,  2097,  2113,  2008,  2009,  2003,  2145,  2006,  1996,
          3200,   102,     0]])
  attention_mask: torch.Size([4, 23])

  Decoded texts from Batch 1:
    Example 1: i didnt feel humiliated
    Example 2: i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake
    Example 3: im grabbing a minute to post i feel greedy wrong
    Example 4: i am ever feeling nostalgic about the fireplace i will know that it is still on the property

  Decoded texts (with special tokens):
    Example 1: [CLS] i didnt feel humiliated [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    Example 2: [CLS] i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake [SEP]
    Example 3: [CLS] im grabbing a minute to post i feel greedy wrong [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    Example 4: [CLS] i am ever feeling nostalgic about the fireplace i will know that it is still on the property [SEP] [PAD]

Batch 2 shapes:
  input_ids: torch.Size([4, 30])
  Input IDs:
tensor([[  101,  1045,  2572,  3110, 24665,  7140, 11714,   102,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        [  101,  4921,  2063,  2042,  3110,  1037,  2210, 10859,  2098,  9906,
          2347,  2102,  2469,  2339,  2008,  2001,   102,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        [  101,  4921,  2063,  2042,  2635,  2030,  4971,  8004,  6444,  2015,
          2030,  2335,  6749,  3815,  1998,  4921,  2063,  5357,  6680,  1037,
          2843,  5514,  2021,  1045,  2036,  2514,  2066,  2061,  6057,   102],
        [  101,  1045,  2514,  2004,  5457,  2055,  2166,  2004,  1037, 10563,
          2030,  2004, 12323,  2094,  2004,  1037,  2095,  2214,  2158,   102,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])
  attention_mask: torch.Size([4, 30])

  Decoded texts from Batch 2:
    Example 1: i am feeling grouchy
    Example 2: ive been feeling a little burdened lately wasnt sure why that was
    Example 3: ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny
    Example 4: i feel as confused about life as a teenager or as jaded as a year old man

  Decoded texts (with special tokens):
    Example 1: [CLS] i am feeling grouchy [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    Example 2: [CLS] ive been feeling a little burdened lately wasnt sure why that was [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    Example 3: [CLS] ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny [SEP]
    Example 4: [CLS] i feel as confused about life as a teenager or as jaded as a year old man [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

## 4-compute_metrics.py

### 4. Evaluation Metrics

Write a function `compute_metrics(predictions)` that calculates key evaluation metrics for a classification model.

The function should:
- Accept an object containing model output logits and true class labels.
- Convert logits to predicted class indices.
- Compute accuracy, precision, recall, and F1-score using scikit-learn.
- Use a weighted average to account for class imbalance in the dataset.
- Return all metrics as a dictionary, ready to be used during model evaluation.

Arguments:
- `predictions`: A `transformers.EvalPrediction` object.

Returns:
- `metrics` (`dict`): Dictionary containing the computed metrics:

```python
{
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float
}
```

```bash
$ cat 4-main.py
#!/usr/bin/env python3

compute_metrics = __import__('4-compute_metrics').compute_metrics
import numpy as np
from types import SimpleNamespace


logits = np.array([
    [2.0, 1.0, 0.1],
    [0.1, 1.5, 0.4],
    [2.0, 0.2, 3.0],
    [0.5, 1.0, 0.3],
    [0.2, 0.1, 2.0]
])

labels = np.array([0, 1, 2, 0, 1])

pred = SimpleNamespace(predictions=logits, label_ids=labels)

metrics = compute_metrics(pred)
print(metrics)

$ ./4-main.py
{'accuracy': 0.6, 'precision': 0.7, 'recall': 0.6, 'f1': 0.5999999999999999}
```

## 5-configure_training_arguments.py

### 5. Configure Training Arguments

Write a function `configure_training_args(output_dir, epochs, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, weight_decay, metric_for_best_model, seed)` that sets up key hyperparameters/arguments for training a Hugging Face model.

The function should also:
- Set default strategies for evaluation and checkpoint saving to `"epoch"`.
- Enable automatic loading of the best checkpoint at the end of training based on the specified metric.
- Specify that higher metric values indicate better performance.
- Ensure the trained model can be pushed to the Hugging Face Hub.

Arguments:
- `output_dir` (`str`): Directory to save checkpoints and logs.
- `epochs` (`int`): Number of training epochs.
- `per_device_train_batch_size` (`int`): Training batch size per device.
- `per_device_eval_batch_size` (`int`): Evaluation batch size per device.
- `learning_rate` (`float`): Learning rate for the optimizer.
- `weight_decay` (`float`): Weight decay for regularization.
- `metric_for_best_model` (`str`): Metric used to determine the best checkpoint.
- `seed` (`int`): Random seed for reproducibility.

Returns:
- `training_args`: A `transformers.TrainingArguments` object ready to be used for model training.

### Test Example

```bash
$ cat 5-main.py
#!/usr/bin/env python3

configure_training_args = __import__('5-configure_training_arguments').configure_training_args

training_args = configure_training_args(
    output_dir="./hbtn_finetuning_emotion_classifier",
    epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    seed=0
)

print(training_args)

$ ./5-main.py
TrainingArguments(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=True,
batch_eval_metrics=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=True,
do_eval=True,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=None,
eval_strategy=IntervalStrategy.EPOCH,
eval_use_gather_object=False,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=True,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_revision=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=no,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=2e-05,
length_column_name=length,
liger_kernel_config=None,
load_best_model_at_end=True,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=./hbtn_finetuning_emotion_classifier/runs/Dec18_11-56-58_367a439cd5ca,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=f1,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3,
optim=OptimizerNames.ADAMW_TORCH_FUSED,
optim_args=None,
optim_target_modules=None,
output_dir=./hbtn_finetuning_emotion_classifier,
overwrite_output_dir=False,
parallelism_config=None,
past_index=-1,
per_device_eval_batch_size=32,
per_device_train_batch_size=16,
prediction_loss_only=False,
project=huggingface,
push_to_hub=True,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['tensorboard', 'wandb'],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=SaveStrategy.EPOCH,
save_total_limit=None,
seed=0,
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
trackio_space_id=trackio,
use_cpu=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.01,
)
```

## 6-training_mode.py

### 6. Training Mode

Write a function `train_model(model, training_args, train_dataset, eval_dataset, test_dataset, tokenizer, data_collator, compute_metrics, model_save_name)` that trains, evaluates, and saves a Hugging Face model.

The function should:
- Fine-tune the provided pre-trained model on the training dataset.
- Evaluate the model on the validation dataset during training.
- Evaluate the model on the test dataset after training.
- Save the trained model and tokenizer to the specified directory.
- Return the Trainer instance, the training results, and the test evaluation results.

Arguments:
- `model` (`transformers.PreTrainedModel`): Pre-trained model to fine-tune.
- `training_args` (`transformers.TrainingArguments`): Training arguments with hyperparameters and strategies.
- `train_dataset` (`datasets.Dataset`): Tokenized training dataset.
- `eval_dataset` (`datasets.Dataset`): Tokenized validation dataset.
- `test_dataset` (`datasets.Dataset`): Tokenized test dataset.
- `tokenizer` (`transformers.PreTrainedTokenizerBase`): Tokenizer (or processing class) used for preprocessing.
- `data_collator` (`callable`): Data collator for dynamic padding.
- `compute_metrics` (`callable`): Function that computes evaluation metrics from predictions and labels.
- `model_save_name` (`str`): Directory to save the trained model and tokenizer.

Returns:
- `trainer` (`transformers.Trainer`): Hugging Face Trainer instance.
- `train_results` (`transformers.Trainer.train()` output): Training metrics and logs.
- `test_results` (`dict`): Evaluation results on the test dataset.

NOTE: Ensure you have a Hugging Face account to push your model there after training. You can follow this guide to learn how to create a Hugging Face API: Hugging Face API Guide

### Launch Model Fine-Tuning

```bash
$ cat 6-main.py
#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map
create_data_collator = __import__('3-dynamic_padding').create_data_collator
compute_metrics = __import__('4-compute_metrics').compute_metrics
configure_training_args = __import__('5-configure_training_arguments').configure_training_args
train_model = __import__('6-training_mode').train_model


from transformers import set_seed
set_seed(0)

import os
from huggingface_hub import login
os.environ["HF_TOKEN"] = "YOUR KEY HERE"
login(token=os.environ["HF_TOKEN"])


dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {name: idx for idx, name in enumerate(label_names)}
id_to_label = {idx: name for idx, name in enumerate(label_names)}
num_classes = len(label_names)
model_name = "distilbert-base-uncased"

tokenizer, model = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=128,
    truncation=True,
    batched=True
)

data_collator = create_data_collator(tokenizer)

training_args = configure_training_args(
    output_dir="./hbtn_emotion_classifier",
    epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    seed=0
)

trainer, train_results, test_results = train_model(
    model=model,
    training_args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    test_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    model_save_name="./hbtn_emotion_classifier_best_model"
)

print("\n Training Results:")
print(f"  Total training time: {train_results.metrics['train_runtime']:.2f} seconds")
print(f"  Training loss: {train_results.training_loss:.4f}")
print(f"  Training samples/sec: {train_results.metrics['train_samples_per_second']:.2f}")
print(f"  Training steps/sec: {train_results.metrics['train_steps_per_second']:.2f}")

print("\n Test Set Performance:")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"  Loss:      {test_results['eval_loss']:.4f}")

print("\n Model and tokenizer saved to:", "./hbtn_emotion_classifier_best_model")
$ ./6-main.py
```

NOTE:

The training of this model was performed using Google Colab, as it provides free access to a GPU, which will significantly reduce training time.

Here are some useful links to guide you on how to use Google Colab for training:

- How to use gpu in google colab?
- Understanding Google Colab Free GPU in detail

### Detailed Test Set Evaluation

```bash
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np


pred_output = trainer.predict(tokenized_test)
pred_labels = np.argmax(pred_output.predictions, axis=-1)
true_labels = pred_output.label_ids
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


conf_matrix = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print("\nPredicted →")
print("Actual ↓")
conf_df = pd.DataFrame(
    conf_matrix,
    index=label_names,
    columns=label_names
)
print(conf_df)


report = classification_report(true_labels, pred_labels, target_names=label_names, digits=4)
print("\nClassification Report:\n", report)
```

NOTE: After completing the training, check the storage space of your Hugging Face Hub account. Your fine-tuned model should be pushed and visible there (see images below).

## 7-inference_mode.py

### 7. Inference Mode

Write a function `inference_mode(model_path, top_k)` that initializes a text classification pipeline using a fine-tuned Hugging Face model.

Arguments:
- `model_path` (`str`): Path to the saved model and tokenizer.
- `top_k` (`int`): Number of top predictions to return per input.

Returns:
- `classifier`: A `transformers.Pipeline` object ready to classify new texts.

### Before Fine-Tuning

```bash
$ cat 7-main_1.py
#!/usr/bin/env python3

# load_distilbert = __import__('1-load_distilbert').load_distilbert
import torch
from transformers import logging, set_seed

set_seed(0)
logging.set_verbosity_error()

model_name = "distilbert-base-uncased"
num_classes = 6

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
label2id = {v: k for k, v in id2label.items()}

tokenizer, model = load_distilbert(model_name, num_classes, id2label, label2id)

texts = [
    "I feel really so disappointed.",
    "I absolutely love spending time with my family.",
    "This is so frustrating! I'm really angry!",
    "I love this new song!",
    "I'm Petrified about what might happen."
]


model.eval()
for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    predicted_emotion = model.config.id2label[predicted_class]

    print(f"Text: {text}")
    print(f"Prediction: {predicted_emotion}")
    print("-" * 40)

$ ./7-main_1.py
Text: I feel really so disappointed.
Prediction: surprise
----------------------------------------
Text: I absolutely love spending time with my family.
Prediction: love
----------------------------------------
Text: This is so frustrating! I'm really angry!
Prediction: anger
----------------------------------------
Text: I love this new song!
Prediction: anger
----------------------------------------
Text: I'm Petrified about what might happen.
Prediction: love
----------------------------------------
```

### After fine tuning with simple vocabulary

```bash
$ cat 7-main_2.py
#!/usr/bin/env python3

inference_mode = __import__('7-inference_mode').inference_mode

model_path = "./hbtn_emotion_classifier_best_model"

texts = [
    "I feel really so disappointed.",
    "I absolutely love spending time with my family.",
    "This is so frustrating! I'm really angry!",
    "I love this new song!",
    "I'm Petrified about what might happen."
]

emotion_classifier = inference_mode(
    model_path,
    top_k=3
)

predictions = emotion_classifier(texts)

for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred}")
    print("-" * 40)

$ ./7-main_2.py
Text: I feel really so disappointed.
Prediction: [{'label': 'sadness', 'score': 0.9996964931488037}, {'label': 'anger', 'score': 9.200986823998392e-05}, {'label': 'joy', 'score': 7.224775617942214e-05}]
----------------------------------------
Text: I absolutely love spending time with my family.
Prediction: [{'label': 'love', 'score': 0.9527739882469177}, {'label': 'joy', 'score': 0.03331461921334267}, {'label': 'anger', 'score': 0.0075605991296470165}]
----------------------------------------
Text: This is so frustrating! I'm really angry!
Prediction: [{'label': 'anger', 'score': 0.9990845918655396}, {'label': 'fear', 'score': 0.00039786173147149384}, {'label': 'sadness', 'score': 0.00018311702297069132}]
----------------------------------------
Text: I love this new song!
Prediction: [{'label': 'joy', 'score': 0.6120102405548096}, {'label': 'anger', 'score': 0.36406224966049194}, {'label': 'love', 'score': 0.010989760980010033}]
----------------------------------------
Text: I'm Petrified about what might happen.
Prediction: [{'label': 'fear', 'score': 0.9990009665489197}, {'label': 'surprise', 'score': 0.0003558320750016719}, {'label': 'anger', 'score': 0.00030839754617772996}]
----------------------------------------
```

### With complex vocabulary

```bash
$ cat 7-main_3.py
#!/usr/bin/env python3

inference_mode = __import__('7-inference_mode').inference_mode

model_path = "./hbtn_emotion_classifier_best_model"

texts = [
    "I am profoundly disenfranchised and disillusioned by the outcome.",
    "I derive immense gratification and elation from familial interactions.",
    "The current state of affairs is entirely exasperating; my indignation is substantial.",
    "This novel musical composition captivates me entirely.",
    "I am utterly paralyzed by an overwhelming apprehension regarding potential eventualities."
]

emotion_classifier = inference_mode(
    model_path,
    top_k=1
)

predictions = emotion_classifier(texts)

for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred}")
    print("-" * 40)

$ ./7-main_3.py
Text: I am profoundly disenfranchised and disillusioned by the outcome.
Prediction: [{'label': 'sadness', 'score': 0.9996447563171387}]
----------------------------------------
Text: I derive immense gratification and elation from familial interactions.
Prediction: [{'label': 'love', 'score': 0.6298142671585083}]
----------------------------------------
Text: The current state of affairs is entirely exasperating; my indignation is substantial.
Prediction: [{'label': 'anger', 'score': 0.998375654220581}]
----------------------------------------
Text: This novel musical composition captivates me entirely.
Prediction: [{'label': 'joy', 'score': 0.9969898462295532}]
----------------------------------------
Text: I am utterly paralyzed by an overwhelming apprehension regarding potential eventualities.
Prediction: [{'label': 'fear', 'score': 0.9917323589324951}]
----------------------------------------
```
