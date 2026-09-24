# LLMs

## 0-load_mlm.py

### 0. (MLM) Load RoBERTa

Write a function `load_mlm(model_name)` that loads a pre-trained RoBERTa model ready for Masked Language Modeling (MLM) using Hugging Face Transformers.

Arguments:
- `model_name` (`str`): Name of the pre-trained model to load.

Returns:
- `model`: An instance of `RobertaForMaskedLM` ready for inference.

### Test Example

```bash
$ cat 0-main.py
#!/usr/bin/env python3

load_mlm = __import__('0-load_mlm').load_mlm

model = load_mlm("roberta-base")
print("Model type:", type(model))

$ ./0-main.py
Model type: <class 'transformers.models.roberta.modeling_roberta.RobertaForMaskedLM'>
```

## 1-tokenization.py

### 1. (MLM) Tokenization

Write a function `tokenize_text(model_name, sentence, padding=True)` that loads a pre-trained RoBERTa tokenizer and tokenizes a given input sentence.

Arguments:
- `model_name` (`str`): Name of the pre-trained model to load.
- `sentence` (`str`): The text to tokenize.
- `padding` (`bool`): Whether to pad the tokenized sequence. Defaults to `True`.

Returns:
- `tokenizer`: An instance of `RobertaTokenizer`.
- `inputs`: Tokenized representation of the sentence as PyTorch tensors.

### Test Example

```bash
$ cat 1-main.py
#!/usr/bin/env python3

tokenize_text = __import__('1-tokenization').tokenize_text

sentences = [
    "AI is dominating the world.",
    "Transformers are the backbone of modern LLMs.",
    "Hello!"
]

tokenizer, inputs = tokenize_text("roberta-base", sentences)

print("Tokenizer type:", type(tokenizer))
print("Input IDs:", inputs['input_ids'])
print("Tokens:")
for i, ids in enumerate(inputs['input_ids']):
    print(f"Sentence {i+1}:", tokenizer.convert_ids_to_tokens(ids))
print("Attention mask:", inputs['attention_mask'])

$ ./1-main.py
Tokenizer type: <class 'transformers.models.roberta.tokenization_roberta.RobertaTokenizer'>
Input IDs: tensor([[    0, 15238,    16, 17349,     5,   232,     4,     2,     1,     1,
             1,     1],
        [    0, 44820,   268,    32,     5, 24456,     9,  2297, 30536, 13123,
             4,     2],
        [    0, 31414,   328,     2,     1,     1,     1,     1,     1,     1,
             1,     1]])
Tokens:
Sentence 1: ['<s>', 'AI', 'Ġis', 'Ġdominating', 'Ġthe', 'Ġworld', '.', '</s>', '<pad>', '<pad>', '<pad>', '<pad>']
Sentence 2: ['<s>', 'Transform', 'ers', 'Ġare', 'Ġthe', 'Ġbackbone', 'Ġof', 'Ġmodern', 'ĠLL', 'Ms', '.', '</s>']
Sentence 3: ['<s>', 'Hello', '!', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
Attention mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
```

## 2-get_mask_index.py

### 2. (MLM) Get Mask Indices

Write a function `get_mask_index(inputs, tokenizer)` that identifies and returns the positions of all `<mask>` tokens in the tokenized input sequence.

The function searches through the input token IDs and extracts the indices where mask tokens appear.

If no mask token is found, the function must raise a `ValueError` with the message: `No <mask> token found in the input!`.

Arguments:
- `inputs`: Tokenized inputs.
- `tokenizer`: An instance of `RobertaTokenizer`.

Returns:
- `mask_indices` (`list[int]`): A list containing the index of every `<mask>` token found in the sequence.

### Test Example

```bash
$ cat 2-main.py
#!/usr/bin/env python3

tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index

sentences = [
    "AI will <mask> the future and <mask> the world.",
    "Transformers are <mask> modern LLMs."
]

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0]

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    mask_indices = get_mask_index(inputs, tokenizer)
    print("Mask indices:", mask_indices)

$ ./2-main.py
Sentence 1: AI will <mask> the future and <mask> the world.
Tokens: ['<s>', 'AI', 'Ġwill', '<mask>', 'Ġthe', 'Ġfuture', 'Ġand', '<mask>', 'Ġthe', 'Ġworld', '.', '</s>']
Token IDs: tensor([    0, 15238,    40, 50264,     5,   499,     8, 50264,     5,   232,
            4,     2])
Mask indices: [3, 7]

Sentence 2: Transformers are <mask> modern LLMs.
Tokens: ['<s>', 'Transform', 'ers', 'Ġare', '<mask>', 'Ġmodern', 'ĠLL', 'Ms', '.', '</s>']
Token IDs: tensor([    0, 44820,   268,    32, 50264,  2297, 30536, 13123,     4,     2])
Mask indices: [4]
```

## 3-compute_mask_logits.py

### 3. (MLM) Compute Mask Logits

Write a function `compute_mask_logits(model, inputs, mask_indices)` that computes the raw logits for all `<mask>` tokens in a tokenized sentence using a pre-trained RoBERTa masked language model.

The function performs a single forward pass through the model and extracts the logits at each mask position.

Gradient computation should be disabled, as this function is intended for inference only.

Arguments:
- `model`: A `RobertaForMaskedLM` model loaded with pre-trained weights.
- `inputs`: Tokenized inputs.
- `mask_indices` (`list[int]`): Positions of all `<mask>` tokens in the input sequence.

Returns:
- `mask_logits_list` (`list[torch.Tensor]`): A list containing logits tensors for each `<mask>` token, representing the model's raw predictions at that masked position.

### Test Example

```bash
$ cat 3-main.py
#!/usr/bin/env python3

load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
    ]

model = load_mlm("roberta-base")

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0]

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    mask_indices = get_mask_index(inputs, tokenizer)
    print("Mask indices:", mask_indices)

    logits_list = compute_mask_logits(model, inputs, mask_indices)
    print(f"Number of masks: {len(logits_list)}")
    print(f"Mask logits: {logits_list}")
    print("Logits shape per mask:", logits_list[0].shape)

$ ./3-main.py
Sentence 1: AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.
Tokens: ['<s>', 'AI', 'Ġcan', '<mask>', 'Ġproductivity', ',', '<mask>', 'Ġcreativity', ',', 'Ġand', 'Ġeven', '<mask>', 'Ġdecision', '-', 'making', '.', '</s>']
Token IDs: tensor([    0, 15238,    64, 50264,  8106,     6, 50264, 11140,     6,     8,
          190, 50264,   568,    12,  5349,     4,     2])
Mask indices: [3, 6, 11]
Number of masks: 3
Logits for all masks found: [tensor([-1.8451, -4.5896,  4.8376,  ..., -3.4534, -1.0701,  1.4316]), tensor([-2.6763, -4.3642,  5.0095,  ..., -2.9155, -0.1945,  1.6553]), tensor([-2.5209, -4.3017,  3.8145,  ..., -2.6969, -1.9371,  1.6832])]
Logits shape for a single mask: torch.Size([50265])

Sentence 2: Machine learning models can <mask> patterns in data and <mask> accurate predictions.
Tokens: ['<s>', 'Machine', 'Ġlearning', 'Ġmodels', 'Ġcan', '<mask>', 'Ġpatterns', 'Ġin', 'Ġdata', 'Ġand', '<mask>', 'Ġaccurate', 'Ġpredictions', '.', '</s>']
Token IDs: tensor([    0, 46100,  2239,  3092,    64, 50264,  8117,    11,   414,     8,
        50264,  6030, 12535,     4,     2])
Mask indices: [5, 10]
Number of masks: 2
Logits for all masks found: [tensor([-2.0629, -4.6536,  4.9012,  ..., -2.8561, -3.2330, -0.0716]), tensor([-5.5869, -4.6707,  5.7269,  ..., -2.4141, -3.6374,  0.3193])]
Logits shape for a single mask: torch.Size([50265])

Sentence 3: Climate change is <mask> the planet and <mask> future generations.
Tokens: ['<s>', 'Climate', 'Ġchange', 'Ġis', '<mask>', 'Ġthe', 'Ġplanet', 'Ġand', '<mask>', 'Ġfuture', 'Ġgenerations', '.', '</s>']
Token IDs: tensor([    0, 40466,   464,    16, 50264,     5,  5518,     8, 50264,   499,
         6808,     4,     2])
Mask indices: [4, 8]
Number of masks: 2
Logits for all masks found: [tensor([ 1.7354, -3.7031,  4.1403,  ..., -1.6459, -0.9220,  2.7873]), tensor([ 1.1310, -3.7601,  5.1612,  ..., -1.1236,  0.6535,  3.8156])]
Logits shape for a single mask: torch.Size([50265])
```

## 4-decode_mask_predictions.py

### 4. (MLM) Decode Mask Predictions

Write a function `decode_mask_predictions(mask_logits_list, tokenizer)` that converts the logits for all `<mask>` tokens into their corresponding vocabulary tokens.

The function maps every token ID in the vocabulary to its string representation, providing a complete decoding that can be used to extract predictions.

Arguments:
- `mask_logits_list` (`list[torch.Tensor]`): A list of logits tensors, one for each `<mask>` token in the sentence.
- `tokenizer`: A RoBERTa tokenizer instance used to decode token IDs into readable strings.

Returns:
- `decoded_tokens` (`list[list[str]]`): A list of lists, where each inner list contains all vocabulary tokens (as strings) corresponding to the logits at each mask position. Each inner list contains one token string for every position in the model's vocabulary.

### Test Example

```bash
$ cat 4-main_1.py
#!/usr/bin/env python3

import torch
load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits
decode_mask_predictions = __import__('4-decode_mask_predictions').decode_mask_predictions

sentence = "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making."

model = load_mlm("roberta-base")

print(f"Original sentence: {sentence}\n")

tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)

mask_indices = get_mask_index(inputs, tokenizer)
print(f"Number of masks found: {len(mask_indices)}\n")

mask_logits_list = compute_mask_logits(model, inputs, mask_indices)

decoded_predictions = decode_mask_predictions(mask_logits_list, tokenizer)

for j, (logits, all_tokens) in enumerate(zip(mask_logits_list, decoded_predictions), 1):
    top10_indices = torch.topk(logits, 10).indices
    top10_logits = logits[top10_indices].tolist()
    top10_tokens = [all_tokens[idx] for idx in top10_indices]

    print(f"Mask {j} - Top 10 predicted tokens with logits:")
    for tkn, logit in zip(top10_tokens, top10_logits):
        print(f"  {tkn:20s}: {logit:.4f}")

$ ./4-main_1.py
Original sentence: AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.

Number of masks found: 3

Mask 1 - Top 10 predicted tokens with logits:
  improve             : 20.7638
  increase            : 20.7551
  boost               : 20.3488
  enhance             : 19.0474
  raise               : 16.1955
  accelerate          : 16.1444
  elevate             : 16.1289
  augment             : 16.0746
  drive               : 16.0712
  maximize            : 15.8619
Mask 2 - Top 10 predicted tokens with logits:
  increase            : 18.1499
  boost               : 17.3278
  improve             : 17.1638
  enhance             : 16.9535
  stimulate           : 16.0096
  foster              : 15.8470
  inspire             : 15.8379
  drive               : 15.7411
  spark               : 14.9729
  fuel                : 14.9697
Mask 3 - Top 10 predicted tokens with logits:
  improve             : 18.6734
  better              : 15.6538
  simplify            : 15.5081
  influence           : 15.4636
  enhance             : 15.4343
  guide               : 15.1359
  aid                 : 15.0812
  optimize            : 14.8491
  drive               : 14.8177
  facilitate          : 14.4888
```

```bash
$ cat 4-main_2.py
#!/usr/bin/env python3

import torch
load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits
decode_mask_predictions = __import__('4-decode_mask_predictions').decode_mask_predictions

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
    ]

model = load_mlm("roberta-base")

for i, sentence in enumerate(sentences, 1):
    print(f"\n{'='*70}")
    print(f"Sentence {i}: {sentence}")
    print('='*70)

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)

    mask_indices = get_mask_index(inputs, tokenizer)

    mask_logits_list = compute_mask_logits(model, inputs, mask_indices)

    decoded_predictions = decode_mask_predictions(mask_logits_list, tokenizer)

    top_predictions = []
    for logits, all_tokens in zip(mask_logits_list, decoded_predictions):
        top_idx = torch.argmax(logits).item()
        top_token = all_tokens[top_idx]
        top_predictions.append(top_token)

    filled_sentence = sentence
    for prediction in top_predictions:
        filled_sentence = filled_sentence.replace(tokenizer.mask_token, prediction, 1)

    print(f"\nPredicted tokens: {top_predictions}")
    print(f"Final sentence: {filled_sentence}")

$ ./4-main_2.py
======================================================================
Sentence 1: AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.
======================================================================

Predicted tokens: ['improve', 'increase', 'improve']
Final sentence: AI can improve productivity, increase creativity, and even improve decision-making.

======================================================================
Sentence 2: Machine learning models can <mask> patterns in data and <mask> accurate predictions.
======================================================================

Predicted tokens: ['detect', 'make']
Final sentence: Machine learning models can detect patterns in data and make accurate predictions.

======================================================================
Sentence 3: Climate change is <mask> the planet and <mask> future generations.
======================================================================

Predicted tokens: ['destroying', 'threatening']
Final sentence: Climate change is destroying the planet and threatening future generations.
```

## 5-fill_mask.py

### 5. (Encoder-Only) Fill Mask

Write a function `fill_mask(model_name, top_k)` that creates a high-level interface for performing Masked Language Modeling using a pre-trained large language model.

This task consolidates all low-level steps from Tasks 0 to 4 into a single high-level interface.

Arguments:
- `model_name` (`str`): Name of the pre-trained model to use.
- `top_k` (`int`): Number of top predictions the pipeline will return for each mask token.

Returns:
- `fill`: A Hugging Face pipeline object.

### Test Example

```bash
$ cat 5-main.py
#!/usr/bin/env python3

from pprint import pprint
fill_mask = __import__('5-fill_mask').fill_mask

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
]

fill = fill_mask(model_name="roberta-base", top_k=1)

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")
    predictions = fill(sentence)
    pprint(predictions)

$ ./5-main.py
Device set to use cpu

Sentence 1: AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.
[[{'score': 0.3345177471637726,
   'sequence': '<s>AI can improve productivity,<mask> creativity, and '
               'even<mask> decision-making.</s>',
   'token': 1477,
   'token_str': ' improve'}],
 [{'score': 0.3293951749801636,
   'sequence': '<s>AI can<mask> productivity, increase creativity, and '
               'even<mask> decision-making.</s>',
   'token': 712,
   'token_str': ' increase'}],
 [{'score': 0.6645009517669678,
   'sequence': '<s>AI can<mask> productivity,<mask> creativity, and even '
               'improve decision-making.</s>',
   'token': 1477,
   'token_str': ' improve'}]]

Sentence 2: Machine learning models can <mask> patterns in data and <mask> accurate predictions.
[[{'score': 0.4116588532924652,
   'sequence': '<s>Machine learning models can detect patterns in data '
               'and<mask> accurate predictions.</s>',
   'token': 10933,
   'token_str': ' detect'}],
 [{'score': 0.7143003940582275,
   'sequence': '<s>Machine learning models can<mask> patterns in data and make '
               'accurate predictions.</s>',
   'token': 146,
   'token_str': ' make'}]]

Sentence 3: Climate change is <mask> the planet and <mask> future generations.
[[{'score': 0.42440468072891235,
   'sequence': '<s>Climate change is destroying the planet and<mask> future '
               'generations.</s>',
   'token': 14340,
   'token_str': ' destroying'}],
 [{'score': 0.551878809928894,
   'sequence': '<s>Climate change is<mask> the planet and threatening future '
               'generations.</s>',
   'token': 5608,
   'token_str': ' threatening'}]]
```

## 6-translate_text.py

### 6. (Encoder-Decoder) Translation

Write a function `translate_text(model_name, src_lang, tgt_lang)` that creates a high-level interface for performing language translation using a pre-trained large language model.

Arguments:
- `model_name` (`str`): Name of the pre-trained model to use.
- `src_lang` (`str`): Source language code (e.g., `"en"` for English).
- `tgt_lang` (`str`): Target language code (e.g., `"fr"` for French).

Returns:
- `translator`: A Hugging Face pipeline object.

### Test Example

```bash
$ cat 6-main_1.py
#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_en_fr = translate_text("Helsinki-NLP/opus-mt-en-fr")
sentence1 = "Artificial intelligence is transforming the world."
sentence2 = "Machine learning models improve over time."

translations = translator_en_fr([sentence1, sentence2])
for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])

$ ./6-main_1.py
Device set to use cpu
Sentence 1 translation: L'intelligence artificielle transforme le monde.
Sentence 2 translation: Les modèles d'apprentissage automatique s'améliorent avec le temps.
```

```bash
$ cat 6-main_2.py
#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_fr_en = translate_text("Helsinki-NLP/opus-mt-fr-en")
sentence1 = "L'intelligence artificielle transforme le monde."
sentence2 = "Les modèles d'apprentissage automatique s'améliorent avec le temps."

translations = translator_fr_en([sentence1, sentence2])
for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])

$ ./6-main_2.py
Device set to use cpu
Sentence 1 translation: Artificial intelligence is transforming the world.
Sentence 2 translation: Machine learning models improve over time.
```

```bash
$ cat 6-main_3.py
#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_mul_en = translate_text("Helsinki-NLP/opus-mt-mul-en")

sentence1 = "L'intelligence artificielle transforme le monde."
sentence2 = "Los modelos de aprendizaje automático mejoran con el tiempo."

translations = translator_mul_en([sentence1, sentence2])

for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])

$ ./6-main_3.py
Device set to use cpu
Sentence 1 translation: Artificial intelligence transforms the world.
Sentence 2 translation: Auto learning models will improve with time.
```

```bash
$ cat 6-main_4.py
#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_fr_en = translate_text("facebook/m2m100_418M", src_lang="fr", tgt_lang="en")
sentence1 = "L'intelligence artificielle transforme le monde."
sentence2 = "Les modèles d'apprentissage automatique s'améliorent avec le temps."

translations = translator_fr_en([sentence1, sentence2])
for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])

$ ./6-main_4.py
Device set to use cpu
Sentence 1 translation: Artificial intelligence is transforming the world.
Sentence 2 translation: Automatic learning models are improving over time.
```

## 7-text_generator.py

### 7. (Decoder-Only) Text Generation

Write a function `create_text_generator(model_name, prompt, max_new_tokens, temperature, repetition_penalty, no_repeat_ngram_size)` that creates a high-level interface for performing text generation using a pre-trained language model, allowing control over the generation behavior.

The function must set the padding token ID (`pad_token_id`) to be the same as the tokenizer end-of-sequence token ID (`eos_token_id`).

Arguments:
- `model_name` (`str`): Name of the pre-trained model to use.
- `prompt` (`str`): The input text to start generation from.
- `max_new_tokens` (`int`): Maximum number of new tokens to generate.
- `temperature` (`float`): Controls randomness of generation.
- `repetition_penalty` (`float`): Penalizes repeated tokens.
- `no_repeat_ngram_size` (`int`): Prevents repeating n-word sequences.

Returns:
- `generator`: A Hugging Face pipeline object.
- `output` (`list[dict]`): List of generated text predictions from the model.

### Test Example

```bash
$ cat 7-main_1.py
#!/usr/bin/env python3

from transformers import set_seed
create_text_generator = __import__('7-text_generator').create_text_generator

set_seed(0)

max_new_tokens = 25
temperature = 0.1
repetition_penalty = 1.2
no_repeat_ngram_size = 3
prompt = "Artificial Intelligence will"

generator, output = create_text_generator("gpt2", prompt, max_new_tokens, 
                                         temperature, repetition_penalty, no_repeat_ngram_size)


print(type(generator))
print("\n")
print(output)

$ ./7-main_1.py
Device set to use cpu
<class 'transformers.pipelines.text_generation.TextGenerationPipeline'>


[{'generated_text': "Artificial Intelligence will be able to learn from the human brain, and it's going to have a lot of potential for making smart decisions.\n"}]
```

```bash
$ cat 7-main_2.py
#!/usr/bin/env python3

from transformers import set_seed
create_text_generator = __import__('7-text_generator').create_text_generator

set_seed(0)

max_new_tokens = 100
temperature = 1
repetition_penalty = 1.3
no_repeat_ngram_size = 2

prompt = "Artificial Intelligence will"
_, output = create_text_generator("gpt2", prompt, max_new_tokens, 
                                         temperature, repetition_penalty, no_repeat_ngram_size)
print(output)

$ ./7-main_2.py
Device set to use cpu
[{'generated_text': 'Artificial Intelligence will always end up with a better user experience, at least in the short run.\n'}]
```

```bash
$ cat 7-main_3.py
#!/usr/bin/env python3

from transformers import set_seed
create_text_generator = __import__('7-text_generator').create_text_generator

set_seed(0)

max_new_tokens = 100
temperature = 1
repetition_penalty = 1.3
no_repeat_ngram_size = 2

prompt = "Artificial Intelligence will"
_, output = create_text_generator("distilgpt2", prompt, max_new_tokens, 
                                         temperature, repetition_penalty, no_repeat_ngram_size)
print(output)

$ ./7-main_3.py
Device set to use cpu
[{'generated_text': 'Artificial Intelligence will be created to help our customers and clients become more transparent about the nature of their data security, while helping them understand how we are managing sensitive information in real time.\n\u200a Follow Me on Twitter'}]
```

## 8-image_classifier.py

### 8. Performing Computer Vision Tasks

Write a function `image_classifier(model)` that creates a high-level interface to perform image classification using a pre-trained large language model adapted for computer-vision applications.

Arguments:
- `model` (`str`): Name of the pre-trained to use.

Returns:
- `classifier`: A Hugging Face pipeline object.

Images used: `images.rar`

### Test Example

```bash
$ cat 8-main_1.py
#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
import pprint
image_classifier = __import__('8-image_classifier').image_classifier

image_path = "dog.jpg"

classification_model = "google/vit-base-patch16-224"

classifier = image_classifier(classification_model)

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

classification_result = classifier(image_path)
pprint.pprint({"Classification result": classification_result})

$ ./8-main_1.py

>> img
>>
{'Classification result': [{'label': 'Labrador retriever',
                            'score': 0.9425420165061951},
                           {'label': 'German short-haired pointer',
                            'score': 0.01687130331993103},
                           {'label': 'flat-coated retriever',
                            'score': 0.01143638789653778},
                           {'label': 'Weimaraner',
                            'score': 0.005267821252346039},
                           {'label': 'Great Dane',
                            'score': 0.004166572354733944}]}
```

```bash
$ cat 8-main_2.py
#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
import pprint
image_classifier = __import__('8-image_classifier').image_classifier

image_path = "animal.PNG"

classification_model = "google/vit-base-patch16-224"

classifier = image_classifier(classification_model)

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

classification_result = classifier(image_path)
pprint.pprint({"Classification result": classification_result})

$ ./8-main_2.py

>> img
>>
{'Classification result': [{'label': 'meerkat, mierkat',
                            'score': 0.8086013793945312},
                           {'label': 'mongoose', 'score': 0.13784733414649963},
                           {'label': 'patas, hussar monkey, Erythrocebus patas',
                            'score': 0.031040875241160393},
                           {'label': 'titi, titi monkey',
                            'score': 0.004792744293808937},
                           {'label': 'squirrel monkey, Saimiri sciureus',
                            'score': 0.0023483172990381718}]}
```

## 9-image_captioner.py

### 9. VLM for Image Captioning

Write a function `image_captioner(model, image_path, max_new_tokens)` that generates a textual description (caption) of a given image using a pre-trained BLIP (Bootstrapped Language-Image Pre-training) Vision-Language Model.

The function should:
- Load the BLIP processor and pre-trained model.
- Convert the image into PyTorch tensors using the processor.
- Generate caption tokens from the model with the processed inputs, specifying the maximum number of tokens.
- Decode the generated tokens into readable text, skipping special tokens.

Arguments:
- `model` (`str`): Name of the pre-trained image captioning model to use.
- `image_path` (`str`): Path to the image file to caption.
- `max_new_tokens` (`int`): Maximum number of tokens to generate.

Returns:
- `caption` (`str`): Generated textual description of the image.

Images used: `images.rar`

### Test Example

```bash
$ cat 9-main_1.py
#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
image_captioner = __import__('9-image_captioner').image_captioner

image_path = "dog.jpg"

caption_model = "Salesforce/blip-image-captioning-base"

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

caption_result = image_captioner(caption_model, image_path, 50)
print("Caption result:", caption_result)

$ ./9-main_1.py

>> img
>> Caption result: a black dog laying on the ground
```

```bash
$ cat 9-main_2.py
#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
image_captioner = __import__('9-image_captioner').image_captioner

image_path = "animal.PNG"

caption_model = "Salesforce/blip-image-captioning-base"

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

caption_result = image_captioner(caption_model, image_path, 10)
print("Caption result:", caption_result)

$ ./9-main_2.py

>> img
>> Caption result: three meerkats on a rock
```
