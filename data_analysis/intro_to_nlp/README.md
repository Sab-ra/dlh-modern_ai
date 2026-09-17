# Intro to NLP

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General:
- What is NLP?
- How to explore text data?
- Why clean and normalize text?
- How to use spaCy vs. NLTK?
- What is tokenization?
- Stemming vs. Lemmatization?
- How to vectorize text?
- Bag-of-Words vs. TF-IDF?
- What are word embeddings?
- How to use Word2Vec/GloVe?
- What is Named Entity Recognition?
- How to analyze sentiment?
- What is topic modeling?
- How to build a text classifier?
- What's next in NLP?

## Prepare Data

### Dataset

The following datasets will be used throughout the tasks:
`./data/SMSSpamCollection.rar`

### Introduction

Before moving on to the tasks, it’s important to note that reading a dataset can sometimes be problematic. Here’s an example:

```bash
$ cat 0-main_1.py 
#!/usr/bin/env python3
import pandas as pd

# 1. default pandas reading
df_default = pd.read_csv('SMSSpamCollection_original', sep='\t',
                         names=['label', 'message'])

print(f"loaded rows with default pandas reading: {len(df_default)}")
print("\nExample of misloaded row:")
print(repr(df_default.iloc[5019]['message']))

$ ./0-main_1.py

loaded rows with default pandas reading: 5507

Example of misloaded row:
'Keep ur problems in ur heart, b\'coz nobody will fight for u. Only u & u have to fight for ur self & win the battle. -VIVEKANAND- G 9t.. SD..\r\nham\tYeah, give me a call if you\'ve got a minute\r\nham\tHI UAWAKE?JUSTFOUND OUT VIA ALETTER THATMUM GOTMARRIED 4thNOV.BEHIND OURBACKS - ANYWAY,I\'L CALL U"'
```

As shown, multiple rows are merged into one due to malformed quotes that break the pandas parsing.

```bash
$ cat 0-main_2.py 
#!/usr/bin/env python3
import csv
import pandas as pd


# 2. pandas reading with quoting=csv.QUOTE_NONE
df = pd.read_csv('SMSSpamCollection_original', sep='\t',
                 names=['label', 'message'],
                 quoting=csv.QUOTE_NONE)
df['message'] = df['message'].str.strip('"')

print(f"loaded rows (quoting=3): {len(df)}")
print(repr(df.iloc[5019]['message']))
print(repr(df.iloc[5020]['message']))
print(repr(df.iloc[5021]['message']))

$ ./0-main_2.py
loaded rows with quoting=3: 5509
"Keep ur problems in ur heart, b'coz nobody will fight for u. Only u & u have to fight for ur self & win the battle. -VIVEKANAND- G 9t.. SD.."
"Yeah, give me a call if you've got a minute"
"HI UAWAKE?JUSTFOUND OUT VIA ALETTER THATMUM GOTMARRIED 4thNOV.BEHIND OURBACKS - ANYWAY,I'L CALL U"
```

The same rows are now correctly split because setting `quoting=csv.QUOTE_NONE` (i.e., quoting=3) disables special handling of quotation marks, treating them as regular characters. The dataset also contains duplicated SMS messages, as shown below:

```bash
$ cat 0-main_3.py
# 3. Duplicates
n_dups = df.duplicated().sum()
print(f"Duplicate rows found      : {n_dups}")

# removing duplicates
df = df.drop_duplicates(ignore_index=True)
print(f"Deduplicated dataset rows: {len(df)}")
df.to_csv('SMSSpamCollection', sep='\t', index=False, header=False)

$ ./0-main_3.py
Duplicate rows found      : 395
Deduplicated dataset rows: 5114
```

After removing duplicates, we save the cleaned dataset as `SMSSpamCollection`, which will be used in all subsequent tasks.

## 0-explore_data.py

### 0. Basic Exploration

Write a function `def explore_data(df):` that performs initial dataset exploration:

Creates a figure with two subplots side by side: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))`

Left subplot: bar chart of ham vs spam counts using `sns.barplot`:

- title: "Ham vs Spam Counts", xlabel: "label", ylabel: "count"

Right subplot: histogram of raw message lengths using `sns.histplot`:

- bins: 50
- title: "Histogram of Raw Message Lengths", xlabel: "length", ylabel: "count"

Uses `plt.tight_layout()`

Returns: `None`

Imports: `import matplotlib.pyplot as plt` and `import seaborn as sns`

### Checker

```python
#!/usr/bin/env python3
import csv
import pandas as pd
explore_data = __import__('0-explore_data').explore_data


df = pd.read_csv('SMSSpamCollection', sep='\t',
                 names=['label', 'message'])
# Basic exploration of the cleaned dataset
print("Dataset Overview:")
print(f"Total messages: {len(df)}\n")

print("Class distribution:")
print((df['label'].value_counts(normalize=True) * 100).round(2))

df['msg_length'] = df['message'].str.len()
print("\nMessage length statistics:")
print(df.groupby('label')['msg_length'].describe())

explore_data(df)

placeholder_mask = df['message'].str.contains(
    r'<#>|<decimal>|<time>|<url>|<email>', regex=True)
print(f"\nDataset placeholders (<#>, <decimal> …): {placeholder_mask.sum()}")

phone_mask = df['message'].str.contains(r'\+?\d[\d\s\-]{6,}\d', regex=True)
print(f"Messages containing phone-like numbers : {phone_mask.sum()}")

url_mask = df['message'].str.contains(r'https?://\S+|www\.\S+', regex=True)
print(f"Messages containing URLs               : {url_mask.sum()}")

currency_mask = df['message'].str.contains(r'[£$€]\d+', regex=True)
print(f"Messages containing currency amounts   : {currency_mask.sum()}")

unicode_mask = df['message'].str.contains(
    r'[\u2018\u2019\u201c\u201d\u2014\u2013\u2026]', regex=True)
print(f"Messages with Unicode punctuation      : {unicode_mask.sum()}")

emoticon_mask = df['message'].str.contains(
    r':\)|:\(|:-\)|:-\(|:D|;\)|<3', regex=True)
print(f"Messages containing ASCII emoticons    : {emoticon_mask.sum()}")

rep_mask = df['message'].str.contains(r'[!?]{2,}', regex=True)
print(f"Messages with repeated punctuations    : {rep_mask.sum()}")
```

## 1-clean_text.py

### 1. Text normalization

Write a function `clean_text(text, replace_num=True, replace_url=True, emoji_action="replace")` that cleans and normalises SMS messages.

Down below is the content of the file `1-clean_text.py`.

```python
#!/usr/bin/env python3
import re
import emoji


_DATASET_PLACEHOLDER_MAP = {
    '<#>':       '<NUM>',
    '<decimal>': '<NUM>',
    '<time>':    '<TIME>',
    '<url>':     '<URL>',
    '<email>':   '<EMAIL>',
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        r"[''‚‛]":    "'",
        r"[""„‟]":    '"',
        r"[‐‑‒–—―−]": "-",
        r"…":          "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    # 1. lowercase + strip
    # 2. dataset placeholders
    # 3. normalize_unicode_punct()
    # 4. URL replacement
    # 5. number replacement (2 passes)
    # 6. emoji handling
    # 7. collapse repeated ! / ?
    # 8. collapse whitespace
```

The function should:

- Return `""` for `None` input. Convert to lowercase and strip leading/trailing whitespace.
- Remap dataset-native placeholders using `_DATASET_PLACEHOLDER_MAP`.
- Convert Unicode punctuation to ASCII with `normalize_unicode_punct()`.
- If `replace_url=True`, replace URLs with `<URL>` using `r'https?://\S+|www\.\S+'`.
- If `replace_num=True`, replace numbers in two passes:
- phone-like strings: `r'\+?\d[\d\s\-]{6,}\d'` (Phone patterns matched first to avoid partial replacement by the general digit pass).
- integers, decimals, currency-prefixed amounts: `r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b'`
- The `(?<!<)` lookbehind prevents matching the digit in emoticons like `<3`.
- Handle emoji using `emoji.replace_emoji()`:
- `emoji_action="replace"` substitutes with `<EMO>`
- `emoji_action="remove"` replaces with a space (not `""` — to prevent token fusion)
- `emoji_action="keep"` leaves untouched
- Collapse runs of repeated `!` or `?` to a single character.
- Collapse any whitespace sequences to a single space and strip.

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
clean_text = __import__('1-clean_text').clean_text

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)

for msg in df[df['message'].str.contains('<#>', regex=False)]['message'].iloc[1:3]:
    print(f"before : {msg}")
    print(f"after  : {clean_text(msg)}\n")

# URL & number replacement
for msg in df[(df['label'] == 'spam') & df['message'].str.contains('http|www', regex=True)]['message'].iloc[:1]:
    print(f"before : {msg[:100]}")
    print(f"after  : {clean_text(msg)[:100]}\n")

for msg in df[(df['label'] == 'spam') & df['message'].str.contains(r'\d{8,}', regex=True)]['message'].iloc[:2]:
    print(f"before : {msg[:100]}")
    print(f"after  : {clean_text(msg)[:100]}\n")

print("Phone patterns matched first to avoid partial replacement by the general digit pass.\n")

# Unicode punctuation — curly quotes and ellipses still present after preclean
for msg in df[df['message'].str.contains('[''…]', regex=True)]['message'].iloc[:2]:
    print(f"before : {msg}")
    print(f"after  : {clean_text(msg)}\n")


# emoji (dataset has only emoticons)
msg = "Hey! 😊 Great offer 🎉 Call now 📞"
for action in ["replace", "remove", "keep"]:
    print(f"[{action:7}] {clean_text(msg, emoji_action=action)}")


df['orig_len']    = df['message'].str.len()
df['cleaned_len'] = df['cleaned'].str.len()
print(df.groupby('label')[['orig_len', 'cleaned_len']].mean().round(1))

```

## 2-tokenize.py

### 2. Tokenization

Write a function `tokenize_text(text, method="tweet")` that tokenizes a cleaned SMS message.

Down below is the content of the file `2-tokenize.py`.

Note: normalize_emoticons() will be discussed later in this task.

```python
#!/usr/bin/env python3
import nltk


EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result

def tokenize_text(text, method="tweet")
    # CODE HERE
```
Arguments:

- `text`: The cleaned SMS message to tokenize.
- `method`: Tokenization strategy to use. Defaults to `"tweet"`.

The function should:

- Return an empty list if `text` is not a string.
- Support the following tokenization methods:
- `"tweet"`: Use NLTK’s `TweetTokenizer` to limit repeated characters to a maximum of 3 consecutive occurrences (e.g., `"loooove" -> "looove"`).
- `"word"`: Use NLTK’s standard punctuation-aware word tokenization.
- `"split"`: Use Python’s basic whitespace splitting.
- Raise `ValueError("Invalid tokenizer method")` if `method` is not supported.
Returns:

- A list of tokens.

Imports: `import nltk`

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
clean_text    = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text


# tweet vs word vs split
msg = clean_text("Don't call me!!! Visit <URL> for FREE prizes... u won :)")
print(f"input : {msg}\n")

for method in ['tweet', 'word', 'split']:
    tokens = tokenize_text(msg, method=method)
    print(f"[{method}] ({len(tokens)}) {tokens}\n")
```

## 3-remove_stopwords.py

### 3. Stopwords Removal

Write a function `remove_stopwords(tokens, language="english", extra_words=None, keep_words=None)` that removes stopwords from a token list.

Arguments:

- `tokens` (`list[str]`): List of tokens to filter.
- `language` (`str`): NLTK stopword language to load. Defaults to `"english"`.
- `extra_words` (`set[str] | None`): Additional words to add to the stopword set.
- `keep_words` (`set[str] | None`): Words to exclude from the stopword set
The function should:

- Return `[]` if `tokens` is not a list.
- Load the NLTK stopword list for the given language.
- Add any words in `extra_words` to the stopword set.
- Remove any words in `keep_words` from the stopword set (e.g. it is used to preserve spam-indicative words that NLTK would otherwise silently discard).
- Return the filtered token list.

Imports: `import nltk`

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text          = __import__('1-clean_text').clean_text
tokenize_text       = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords    = __import__('3-remove_stopwords').remove_stopwords

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens']  = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']  = df['tokens'].apply(normalize_emoticons)
spam_keep_words = {"won", "our", "from", "now", "your", "only"}

# NLTK blindly removes spam-signal words but sometimes some of these words
# could be spam indicative so we need to first look for commun words for both spam and ham messages 
spam_kw = df[
    (df['label'] == 'spam') &
    df['tokens'].apply(lambda t: any(w in t for w in spam_keep_words))
]['tokens'].iloc[3]

print(f"spam tokens         : {spam_kw[:15]}")
print(f"without keep_words  : {remove_stopwords(spam_kw[:15])}")
print(f"with SPAM_KEEP_WORDS: {remove_stopwords(spam_kw[:15], keep_words=spam_keep_words)}")
rescued = sorted(set(remove_stopwords(spam_kw[:15], keep_words=spam_keep_words)) -
                 set(remove_stopwords(spam_kw[:15])))
print(f"rescued             : {rescued}\n")

df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words)
)

tok_before = sum(len(t) for t in df['tokens'])
tok_after  = sum(len(t) for t in df['tokens_no_stop'])
voc_before = set(t for tok in df['tokens']         for t in tok)
voc_after  = set(t for tok in df['tokens_no_stop'] for t in tok)
print(f"tokens : {tok_before:,} -> {tok_after:,}  ({(1-tok_after/tok_before)*100:.1f}% reduction)")
print(f"vocab  : {len(voc_before):,} -> {len(voc_after):,}  ({(1-len(voc_after)/len(voc_before))*100:.1f}% reduction)\n")

removed = Counter(t for tok in df['tokens'] for t in tok) \
        - Counter(t for tok in df['tokens_no_stop'] for t in tok)
print("top removed:")
for w, c in removed.most_common(10):
    print(f"  {w:<12} {c}")
```

## 4-filter_tokens.py

### 4. Filtering

Write a function `filter_tokens(tokens, min_len=2, strip_hashtag=False)` that removes low-information tokens.

Arguments:

- `tokens` (`list[str]`): List of tokens to filter.
- `min_len` (`int`): Minimum token length to keep.
- `strip_hashtag` (`bool`): If True, removes leading `#` from hashtags before processing.
The function should:

- Return `[]` for an empty or falsy `tokens` input.
- Keep any token that matches `_PLACEHOLDER_RE`.
- If `strip_hashtag=True` and the token starts with `#`, strip the `#` prefix before continuing (e.g. `"#free" -> "free"`).
- Drop tokens shorter than `min_len`.
- Drop tokens that contain no alphabetic characters.
- Imports: `import re`
- Before the function, define `PLACEHOLDERRE = re.compile(r'^<[A-Za-z]+>$')`, a compiled regex that strictly matches pipeline placeholders (`<NUM>`, `<URL>`, `<EMO>`, `<TIME>`, `<EMAIL>`). Deliberately strict `<3`, `</3`, and `<word` do not match.

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))

for idx in [2, 8, 100]:
    before = df.iloc[idx]['tokens_no_stop']
    after = filter_tokens(before)
    dropped = sorted(set(before) - set(after))
    print(f"[{df.iloc[idx]['label']}]")
    print(f"  before  : {before[:12]}")
    print(f"  after   : {after[:12]}")
    print(f"  dropped : {dropped}\n")


df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)

for step, col in [("tokenize", 'tokens'), ("stopwords", 'tokens_no_stop'),
                  ("filter", 'tokens_filtered')]:
    total = sum(len(t) for t in df[col])
    unique = len(set(t for tok in df[col] for t in tok))
    print(f"after {step:<12} : {total:>6} tokens  {unique:>5} unique")

filtered_set = set(t for tok in df['tokens_filtered'] for t in tok)
dropped_types = Counter(
    t for tok in df['tokens_no_stop'] for t in tok if t not in filtered_set
)
print("\ntop dropped:")
for w, c in dropped_types.most_common(10):
    print(f"  {w:<12} {c}")
```

## 5-normalize_tokens.py

### 5. Lemma vs Stem

Write a function `normalize_tokens(tokens, method="lemmatize")` that normalises tokens via lemmatisation or stemming.

Arguments:

- `tokens` (`list[str]`): List of tokens to normalize.
- `method` (`str`): Normalization method. Must be `"lemmatize"` or `"stem"`.
The function should:

- Raise `ValueError("method must be 'lemmatize' or 'stem'")` for any other value.
- For `method="stem"`: apply `PorterStemmer` to each token, skipping placeholders.
- For `method="lemmatize"`: apply POS-aware lemmatisation using `WordNetLemmatizer`.
- First tag the tokens with `nltk.pos_tag()`
- then lemmatise each token using its mapped POS from `get_pos()`
- Imports: `import nltk` and `import re`

Note: skipping placeholders. This produces more accurate results than assuming every token is a noun.

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
import nltk
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)

test_tokens = ['running', 'runs', 'ran', 'runner', 'winning', 'won', 'called', 'calling']
print(f"input     : {test_tokens}")
print(f"lemmatize : {normalize_tokens(test_tokens, method='lemmatize')}")
print(f"stem      : {normalize_tokens(test_tokens, method='stem')}\n")

lemmatizer = nltk.stem.WordNetLemmatizer()
for tokens in [["won", "winning"], ["running"], ["better"]]:
    naive = [lemmatizer.lemmatize(t) for t in tokens]
    pos_aw = normalize_tokens(tokens, method="lemmatize")
    for t, n, p in zip(tokens, naive, pos_aw):
        print(f"{t:<10} naive={n:<10} pos-aware={p}")

df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)
df['tokens_stem']  = df['tokens_filtered'].apply(lambda t: normalize_tokens(t, method='stem'))

for label, col in [("filtered", 'tokens_filtered'), ("lemmatize", 'tokens_lemma'), ("stem", 'tokens_stem')]:
    print(f"{label:<12} : {len(set(t for tok in df[col] for t in tok)):>5} unique tokens")

for cls in ['spam', 'ham']:
    counts = Counter(t for tok in df[df.label == cls]['tokens_lemma'] for t in tok)
    print(f"top [{cls}] : {[w for w, _ in counts.most_common(10)]}")

$ ./5-main.py
input     : ['running', 'runs', 'ran', 'runner', 'winning', 'won', 'called', 'calling']
lemmatize : ['run', 'run', 'run', 'runner', 'win', 'win', 'call', 'call']
stem      : ['run', 'run', 'ran', 'runner', 'win', 'won', 'call', 'call']
```

## 6-ngram.py

### 6. N-gram

Write a function `generate_ngrams(tokens, n=2)` that generates n-grams from a token list.

Arguments:

- `tokens` (`list[str]`): List of tokens used to generate n-grams.
- `n` (`int`): Size of each n-gram.
The function should:

- Return `[]` if `tokens` is not a list or has fewer than `n` elements.
- Use `nltk` to generate n-grams.
- Return a list of strings where each n-gram is `n` consecutive tokens joined with `"_"`.

Imports: `import nltk`

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
generate_ngrams = __import__('6-ngram').generate_ngrams

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)
df['bigrams'] = df['tokens_lemma'].apply(lambda t: generate_ngrams(t, n=2))

# n=2 : bigrams capture patterns invisible at the unigram level
spam_tok = df[df['label'] == 'spam']['tokens_lemma'].iloc[0]
for n in [2, 3]:
    print(f"{n}-grams : {generate_ngrams(spam_tok, n=n)[:4]}")
print()

spam_bi = Counter(g for ngs in df[df.label == 'spam']['bigrams'] for g in ngs)
ham_bi = Counter(g for ngs in df[df.label == 'ham']['bigrams'] for g in ngs)

print(f"{'top spam bigrams':<32} {'top ham bigrams'}")
for (sg, sc), (hg, hc) in zip(spam_bi.most_common(12), ham_bi.most_common(12)):
    print(f"  {sg:<28} ({sc:>3})   {hg:<22} ({hc:>3})")

print(f"\n{'bigram':<26} {'spam':>6} {'ham':>6} {'ratio':>8}")
disc = sorted(
    [(ng, sc, ham_bi[ng], sc / max(ham_bi[ng], 1))
     for ng, sc in spam_bi.items() if sc >= 5],
    key=lambda x: -x[3]
)
for ng, sc, hc, ratio in disc[:10]:
    print(f"  {ng:<24} {sc:>6} {hc:>6} {ratio:>7.1f}x")


$ ./6-main.py
2-grams : ['free_entry', 'entry_<NUM>', '<NUM>_wkly', 'wkly_comp']
3-grams : ['free_entry_<NUM>', 'entry_<NUM>_wkly', '<NUM>_wkly_comp', 'wkly_comp_win']

top spam bigrams                 top ham bigrams
  call_<NUM>                   (209)   <NUM>_<NUM>            ( 66)
  <NUM>_<NUM>                  (132)   let_know               ( 42)
  <NUM>_now                    ( 66)   go_<NUM>               ( 40)
  win_<NUM>                    ( 58)   take_care              ( 32)
  your_mobile                  ( 45)   wan_<NUM>              ( 31)
  <NUM>_prize                  ( 44)   new_year               ( 28)
  <NUM>_cash                   ( 44)   like_<NUM>             ( 27)
  please_call                  ( 42)   <NUM>_min              ( 27)
  <NUM>_free                   ( 38)   right_now              ( 24)
  <NUM>_claim                  ( 36)   good_morning           ( 23)
  <NUM>_from                   ( 35)   <NUM>_go               ( 22)
  now_<NUM>                    ( 31)   get_<NUM>              ( 21)

bigram                       spam    ham    ratio
  <NUM>_now                    66      1    66.0x
  win_<NUM>                    58      0    58.0x
  your_mobile                  45      0    45.0x
  <NUM>_prize                  44      0    44.0x
  <NUM>_cash                   44      0    44.0x
  <NUM>_free                   38      1    38.0x
  <NUM>_claim                  36      0    36.0x
  call_<NUM>                  209      7    29.9x
  your_<NUM>                   27      0    27.0x
  po_box                       26      1    26.0x
```

## 7-freq.py

### 7. Word Frequency Distribution

Write a function `def plot_top_n_frequencies(corpus_tokens, n=20):` that plots the most frequent tokens in a preprocessed corpus.

Arguments:

- `corpus_tokens` (`list[list[str]]`): Corpus represented as a list of token lists.
- `n` (`int`): Number of top frequent tokens to display.
The Function should:

- Flatten the list of lists into a single token list and computes frequencies with `nltk.FreqDist`.
- Plot a bar chart using `plt.bar`:
- `figsize=(12, 5)`
- x-tick labels rotated 45° with `ha="right"`
- title: `f"Top {n} Most Frequent Words"`, xlabel: `"Word"`, ylabel: `"Frequency"`
- call `plt.tight_layout()`
- Return the full frequency distribution object.

Imports: `import nltk` and `import matplotlib.pyplot as plt`

### Checker

```python
#!/usr/bin/env python3
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
plot_top_n_frequencies = __import__('7-freq').plot_top_n_frequencies

spam_keep_words = {"our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])


# Full preprocessing pipeline
def preprocess(msg):
    t = tokenize_text(clean_text(msg), method="tweet")
    t = normalize_emoticons(t)
    t = remove_stopwords(t, keep_words=spam_keep_words)
    t = filter_tokens(t)
    t = normalize_tokens(t, method="stem")
    return t


df['tokens'] = df['message'].apply(preprocess)

# top-20 frequency distribution
fd_all = plot_top_n_frequencies(df['tokens'].tolist(), n=20)

# Per-class frequency distributions
spam_tokens = df[df['label'] == 'spam']['tokens'].tolist()
ham_tokens = df[df['label'] == 'ham']['tokens'].tolist()

print("\nTop 15 words in SPAM:")
fd_spam = plot_top_n_frequencies(spam_tokens, n=15)

print("\nTop 15 words in HAM:")
fd_ham = plot_top_n_frequencies(ham_tokens, n=15)

# Distinctive spam words
spam_total_tokens = sum(len(doc) for doc in spam_tokens)
ham_total_tokens = sum(len(doc) for doc in ham_tokens)

spam_unique = {}
for word, spam_count in fd_spam.most_common(50):
    ham_count = fd_ham.get(word, 0)
    spam_rate = spam_count / spam_total_tokens
    ham_rate = ham_count / ham_total_tokens if ham_count > 0 else 0.0001
    ratio = spam_rate / ham_rate
    if ratio > 5:
        spam_unique[word] = ratio

print("\nDISTINCTIVE SPAM WORDS:")
for word, ratio in sorted(spam_unique.items(),
                          key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {word:15}: {ratio:6.1f}x more frequent in spam")
```

## 8-wordcloud.py

### 8. Wordclouds

Write a function `def generate_wordcloud(corpus_tokens, max_words=200, label=None):` that generates a word cloud from a preprocessed corpus.

Arguments:

- `corpus_tokens` (`list[list[str]]`): Corpus represented as a list of token lists.
- `max_words` (`int`): Maximum number of words in the word cloud.
- `label` (`str | None`): Optional title label for the plot.
The function should:

- Concatenate all tokens into a single string.
- Create a `WordCloud` with these exact parameters:
- `max_words=max_words, background_color="white", width=800, height=400, random_state=42`
- Display the word cloud:
- `figsize=(10, 5)`
- `plt.imshow(wc, interpolation="bilinear")`
- `plt.axis("off")`
- title: `f"WordCloud — {label}"` if label is provided, otherwise `"WordCloud"`
- Use `tight_layout`
- Return: the fitted wordcloud object.
- Imports: `import wordcloud` and `import matplotlib.pyplot as plt`

Note: WordCloud's internal tokenizer splits on whitespace and strips punctuation, so pipeline placeholders like `<NUM>` are rendered as `num` and `<URL>` as `url`.

### Checker

```python
#!/usr/bin/env python3
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
generate_wordcloud = __import__('8-wordcloud').generate_wordcloud

spam_keep_words = {"our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])


# Full preprocessing pipeline
def preprocess(msg):
    t = tokenize_text(clean_text(msg), method="tweet")
    t = normalize_emoticons(t)
    t = remove_stopwords(t, keep_words=spam_keep_words)
    t = filter_tokens(t)
    t = normalize_tokens(t, method="stem")
    return t


df['tokens'] = df['message'].apply(preprocess)

spam_tokens = df[df['label'] == 'spam']['tokens'].tolist()
ham_tokens = df[df['label'] == 'ham']['tokens'].tolist()

# Word clouds
wc_spam = generate_wordcloud(spam_tokens, max_words=100, label='SPAM')
wc_ham = generate_wordcloud(ham_tokens,  max_words=100, label='HAM')
wc_all = generate_wordcloud(df['tokens'].tolist(), max_words=150)

# Top weighted words per cloud
print("\nSPAM word cloud:")
for word, weight in sorted(wc_spam.words_.items(),
                           key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {word:15}: weight {weight:.4f}")

print("\nHAM word cloud:")
for word, weight in sorted(wc_ham.words_.items(),
                           key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {word:15}: weight {weight:.4f}")

```

## 10-bow.py

### 9. BoW

Write a function `bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, binary=False)` that builds a Bag-of-Words feature matrix from a list of token lists.

The function should:

- Join each token list into a whitespace-separated string (`CountVectorizer` works on strings).

Use `sklearn` with:

- `tokenizer=str.split` and `lowercase=False` since tokens are already lowercased by the pipeline.

- `token_pattern=None` required when tokenizer is overridden.

Note: The remaining parameters passed through as-is.

Return (X, vectorizer) where:

- `X`: the sparse feature matrix (n_samples, n_features).
- `vectorizer`: the fitted `CountVectorizer` object.

Imports: `import sklearn`

### Checker

```python
#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
bag_of_words = __import__('10-bag_of_words').bag_of_words

spam_keep_words = {"won", "our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()
labels = df['label'].tolist()

for ngram in [(1, 1), (1, 2)]:
    X, vect = bag_of_words(corpus, ngram_range=ngram)
    print(f"ngram_range={ngram} : shape={X.shape}  vocab={len(vect.vocabulary_)}")

X, vect = bag_of_words(corpus)
features   = np.array(vect.get_feature_names_out())
spam_mask  = np.array(labels) == 'spam'
spam_sums  = np.asarray(X[spam_mask].sum(axis=0)).flatten()
top_spam   = features[spam_sums.argsort()[::-1][:23]]
print(f"top spam features  : {list(top_spam)}\n")

ham_mask  = ~spam_mask
ham_sums  = np.asarray(X[ham_mask].sum(axis=0)).flatten()
top_ham   = features[ham_sums.argsort()[::-1][:23]]
print(f"top ham  features  : {list(top_ham)}\n")

X_count,  _ = bag_of_words(corpus, binary=False)
X_binary, _ = bag_of_words(corpus, binary=True)
print(f"count  matrix: max value in a cell : {X_count.max()}")
print(f"binary matrix: max value in a cell : {X_binary.max()}")

```

## 11-tf_idf.py

### 10. TF-IDF

Write a function `tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, norm='l2')` that builds a TF-IDF feature matrix from a list of token lists.

The function should:

- Join each token list into a whitespace-separated string.
- Use `sklearn` with `tokenizer=str.split`, `lowercase=False`, `token_pattern=None`, and the remaining parameters passed through.
Return (X, vectorizer) where:
- `X` is the sparse TF-IDF feature matrix (n_samples, n_features).
- `vectorizer` is the fitted `TfidfVectorizer` object.

Imports: `import sklearn`

### Checker

```python
#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text          = __import__('1-clean_text').clean_text
tokenize_text       = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords    = __import__('3-remove_stopwords').remove_stopwords
filter_tokens       = __import__('4-filter_tokens').filter_tokens
normalize_tokens    = __import__('5-normalize_tokens').normalize_tokens
bag_of_words        = __import__('10-bag_of_words').bag_of_words
tf_idf              = __import__('11-tf_idf').tf_idf

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()
labels = np.array(df['label'].tolist())
spam_mask = labels == 'spam'

X_bow,  v_bow  = bag_of_words(corpus)
X_tf,   v_tf   = tf_idf(corpus)

print(f"BoW shape    : {X_bow.shape}  dtype={X_bow.dtype}")
print(f"TF-IDF shape : {X_tf.shape}  dtype={X_tf.dtype}\n")

features_tf = np.array(v_tf.get_feature_names_out())
features_bw = np.array(v_bow.get_feature_names_out())

spam_tfidf_sum = np.asarray(X_tf[spam_mask].sum(axis=0)).flatten()
top_tfidf = features_tf[spam_tfidf_sum.argsort()[::-1][:15]]
print(f"top spam TF-IDF features : {list(top_tfidf)}\n")

idf = v_tf.idf_
low_idf_idx = idf.argsort()[:10]
print("lowest IDF (most common terms):")
for i in low_idf_idx:
    print(f"  {features_tf[i]:<25} idf={idf[i]:.3f}")

```

```bash
$ ./11-main.py
BoW shape    : (5114, 5000)  dtype=int64
TF-IDF shape : (5114, 5000)  dtype=float64

top spam TF-IDF features : ['<NUM>', 'call', 'call <NUM>', 'your', 'free', '<NUM> <NUM>', 'now', 'mobile', 'win', 'txt', 'claim', 'from', 'text', 'reply', '<URL>']

lowest IDF (most common terms):
  <NUM>                     idf=2.375
  get                       idf=3.153
  your                      idf=3.296
  call                      idf=3.325
  go                        idf=3.379
  now                       idf=3.502
  <EMO>                     idf=3.787
  come                      idf=3.884
  ur                        idf=3.999
  from                      idf=4.003
```

_IDF downweights terms that appear in many documents (common across both classes). High-IDF spam terms are the most discriminative features (i.e. they are specific enough to one class that a classifier can rely on them)._

## 12-word2vec.py

### 11. Word2Vec

Write a function `word2vec_embeddings(corpus_tokens, vector_size=100, window=5, min_count=2, sg=0, epochs=10, workers=4)` that trains Word2Vec and returns per-message embeddings. The function should:

- Train `gensim Word2Vec` on `corpus_tokens`.
- Represent each message as the mean of its in-vocab token vectors. Tokens not in the `Word2Vec` vocabulary are silently ignored. If a message has no in-vocab tokens, its row is a zero vector.
- Return (X, model) where:
- `X` is a `np.ndarray` of shape (n_messages, vector_size).
- `model` is the trained `Word2Vec` model.

Imports: `import numpy as np` and `import gensim.models`

### Checker

```python
#!/usr/bin/env python3

import pandas as pd
clean_text           = __import__('1-clean_text').clean_text
tokenize_text        = __import__('2-tokenize').tokenize_text
normalize_emoticons  = __import__('2-tokenize').normalize_emoticons
remove_stopwords     = __import__('3-remove_stopwords').remove_stopwords
filter_tokens        = __import__('4-filter_tokens').filter_tokens
normalize_tokens     = __import__('5-normalize_tokens').normalize_tokens
word2vec_embeddings  = __import__('12-word2vec').word2vec_embeddings

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()

X, model = word2vec_embeddings(corpus)

print(f"embedding matrix : {X.shape}  (one row per message)\n")

#  semantic relationships learned through the corpus 
for word in ['free', 'call', 'win', 'go']:
    if word in model.wv:
        similar = [w for w, _ in model.wv.most_similar(word, topn=5)]
        print(f"most similar to '{word}' : {similar}")

# OOV
oov_token = "xyzunseen"
in_vocab  = oov_token in model.wv
print(f"'{oov_token}' in vocab : {in_vocab}")
print(f"'free' in vocab      : {'free' in model.wv}\n")

# zero-vector messages
import numpy as np
zero_rows = (X == 0).all(axis=1).sum()
print(f"zero-vector messages : {zero_rows} / {len(X)}")
print(f"avg embedding norm   : {np.linalg.norm(X, axis=1).mean():.4f}")

```

## 13-fasttext.py

### 12. FastText

Write a function `fasttext_embeddings(corpus_tokens, vector_size=100, window=5, min_count=1, sg=0, epochs=10, workers=4)` that trains FastText and returns per-message embeddings.

The function should:

- Train `gensim FastText` on `corpus_tokens`.

- Represent each message as the mean of its token vectors. Unlike `Word2Vec`, `FastText` uses subword n-grams, so every token has a vector including OOV tokens. `min_count=1` is appropriate here for the same reason.

Return (X, model) where:

- `X` is a `np.ndarray` of shape (n_messages, vector_size).
- `model` is the trained `FastText` model.

Imports: `import numpy as np` and `import gensim.models`

### Checker

```python
#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text           = __import__('1-clean_text').clean_text
tokenize_text        = __import__('2-tokenize').tokenize_text
normalize_emoticons  = __import__('2-tokenize').normalize_emoticons
remove_stopwords     = __import__('3-remove_stopwords').remove_stopwords
filter_tokens        = __import__('4-filter_tokens').filter_tokens
normalize_tokens     = __import__('5-normalize_tokens').normalize_tokens
word2vec_embeddings  = __import__('12-word2vec').word2vec_embeddings
fasttext_embeddings  = __import__('13-fasttext').fasttext_embeddings

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()

X_w2v, m_w2v = word2vec_embeddings(corpus)
X_ft,  m_ft  = fasttext_embeddings(corpus)

print(f"Word2Vec  : {X_w2v.shape}")
print(f"FastText  : {X_ft.shape}\n")

# OOV handling
oov_tokens = ['freee', 'calll', 'prze', 'wnnr']
print(f"{'token':<12} {'in W2V vocab':>14} {'FastText has vector':>20}")
print("-" * 50)
for t in oov_tokens:
    in_w2v = t in m_w2v.wv
    print(f"  {t:<10} {str(in_w2v):>14} {'True':>20}")

# zero-vector messages
w2v_zeros = (X_w2v == 0).all(axis=1).sum()
ft_zeros  = (X_ft  == 0).all(axis=1).sum()
print(f"Word2Vec  zero-vector messages : {w2v_zeros}")
print(f"FastText  zero-vector messages : {ft_zeros}\n")

# similar words
for word in ['free', 'call', 'win']:
    if word in m_ft.wv:
        similar = [w for w, _ in m_ft.wv.most_similar(word, topn=5)]
        print(f"FastText most similar to '{word}' : {similar}")

```
