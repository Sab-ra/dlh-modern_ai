# Intro to NLP — Learning Plan

Study path for the school tasks in this folder.
The assignment spec stays in [README.md](README.md). Do not rewrite it.

This plan is how to *learn* the pipeline, not a dump of checker-passing
code. Extra objectives (spaCy, GloVe, NER, sentiment, topics, a
classifier) are study notes only. No new `14-*.py` tasks.

`*.ipynb` is gitignored in this repo. Notebooks here are local only.

Style for later scripts (when you write them): shebang, module and
function docstrings, 79-char lines, `pycodestyle==2.14.0`. Minimal
comments. Plot what the NLP README asks (seaborn in task 0 is fine).

---

## Mental model

SMS spam/ham is a tiny language problem with ugly text: phones, URLs,
emoticons, SMS slang, class imbalance.

Pipeline:

```text
.rar archive
  -> load with quoting=csv.QUOTE_NONE
  -> drop 395 duplicate rows  (5114 left)
  -> clean / normalize
  -> tokenize (tweet tokenizer default)
  -> map emoticons
  -> stopwords, with a spam keep-list
  -> filter junk tokens (keep <NUM>-style placeholders)
  -> lemmatize or stem
  -> n-grams / frequencies / wordclouds
  -> BoW / TF-IDF / mean Word2Vec / mean FastText
```

If you skip a stage, later checkers import it and die. Sequential is
not optional.

---

## Dataset (do this before task 0)

- Archive: [data/SMSSpamCollection.rar](data/SMSSpamCollection.rar)
- Checkers read `SMSSpamCollection` from **this directory's CWD**
  (tab-separated, no header, columns `label`, `message`).

Default `pd.read_csv` merges rows because of malformed quotes.
Use `quoting=csv.QUOTE_NONE`, strip leftover `"`, then
`drop_duplicates(ignore_index=True)`.

Expected: 5509 loaded, 395 duplicates, **5114** remaining.

NLTK data (once, not inside graded functions): `punkt` / `punkt_tab`,
`stopwords`, `wordnet`, `omw-1.4`, POS tagger (`averaged_perceptron_tagger`
and `_eng` if 3.9 complains).

Pinned in repo `requirements.txt`: nltk, emoji, gensim, wordcloud,
sklearn, seaborn, matplotlib. **spaCy is not installed. Do not add it.**

---

## Track A — School tasks (stay on the program)

Filename source of truth is the checker `__import__`, not the heading.
There is **no task 9**. BoW file is `10-bag_of_words.py`, not `10-bow.py`.

| Order | File | Concept you must be able to explain |
|------:|------|-------------------------------------|
| 0 | `0-explore_data.py` | What the corpus looks like before you touch it |
| 1 | `1-clean_text.py` | Why raw SMS is unusable; placeholders vs real tokens |
| 2 | `2-tokenize.py` | What a token is; tweet vs word vs split |
| 3 | `3-remove_stopwords.py` | Stopwords can delete the signal |
| 4 | `4-filter_tokens.py` | Short / non-alpha noise vs kept `<NUM>` |
| 5 | `5-normalize_tokens.py` | Stem (crude) vs POS-aware lemma |
| 6 | `6-ngram.py` | Phrases you cannot see as unigrams |
| 7 | `7-freq.py` | Corpus-level counts |
| 8 | `8-wordcloud.py` | Same counts, prettier, less precise |
| 9 | `10-bag_of_words.py` | Sparse count vectors |
| 10 | `11-tf_idf.py` | Downweight terms that appear everywhere |
| 11 | `12-word2vec.py` | Dense embeddings; OOV = no vector |
| 12 | `13-fasttext.py` | Subword embeddings; OOV still has a vector |

Write one function per file. Run that task's checker from this folder
before moving on. Later checkers chain `clean_text` → … → embeddings.

### 0. Explore

`explore_data(df)`: two subplots `(1, 2)`, `figsize=(12, 4)`.

- Left: `sns.barplot` ham vs spam counts. Title `"Ham vs Spam Counts"`.
- Right: `sns.histplot` of raw `message` length, `bins=50`.
- `plt.tight_layout()`, return `None`.

Also look (checker prints this): class %, length by label, phones, URLs,
currency, unicode punct, emoticons, repeated `!?`. Imbalance is the
diagnosis; length and phones are symptoms.

### 1. Clean

`clean_text(text, replace_num=True, replace_url=True, emoji_action="replace")`

Order matters: lowercase+strip → dataset placeholders → unicode punct →
URL → **phone regex before general digits** → emoji → collapse `!`/`?` →
whitespace.

- `None` → `""`
- `emoji_action="remove"` replaces with a **space**, not `""`
- Phone first so `+447123456789` does not become a pile of `<NUM>` scraps

### 2. Tokenize

`tokenize_text(text, method="tweet")` plus `normalize_emoticons`.

- non-string → `[]`
- `"tweet"`: `TweetTokenizer(reduce_len=True)` (`loooove` → `looove`)
- `"word"`: NLTK word tokenizer (splits `Don't`, eats some emoticons)
- `"split"`: whitespace only
- else: `ValueError("Invalid tokenizer method")`

Tweet tokenizer is the default because SMS is closer to Twitter than
to Jane Austen.

### 3. Stopwords

NLTK English list, plus `extra_words`, minus `keep_words`.

Spam keep-list used in checkers: `won`, `our`, `from`, `now`, `your`,
`only`. Blind stopword removal deletes “you have won”. That is the
whole lesson.

Non-list → `[]`.

### 4. Filter

Keep tokens matching `r'^<[A-Za-z]+>$'` (`<NUM>`, `<URL>`, `<EMO>`,
`<TIME>`, `<EMAIL>`). Not `<3`.

README names both `_PLACEHOLDER_RE` and `PLACEHOLDERRE`. Define one
compiled regex and alias the other.

Drop `len < min_len` and tokens with no alphabetic character.
Optional `strip_hashtag=True`: `#free` → `free`.

### 5. Lemma vs stem

`normalize_tokens(tokens, method="lemmatize")`

- stem: Porter, skip placeholders
- lemmatize: `pos_tag` then WordNet via `get_pos()` — not “everything
  is a noun”
- else: `ValueError("method must be 'lemmatize' or 'stem'")`

Isolated lists like `['won']` can tag as noun. If the checker printout
disagrees with naive lemma, POS mapping is the usual suspect.

### 6. N-grams

`nltk.ngrams`, join with `"_"`. Too short a list → `[]`.

Spam bigrams (`call_<NUM>`, `win_<NUM>`) are the punchline: unigrams
lie by omission.

### 7–8. Frequency and wordclouds

- Freq: flatten, `nltk.FreqDist`, `plt.bar`, figsize `(12, 5)`, ticks
  45° `ha="right"`, return the FreqDist.
- WordCloud: exact kwargs (`white`, 800×400, `random_state=42`).
  WordCloud strips `<>`, so `<NUM>` renders as `num`.

Pretty pictures are not features. They are a sanity check.

### 9–10. BoW and TF-IDF

Join each token list to a string. Vectorizer kwargs:

- `tokenizer=str.split`, `lowercase=False`, `token_pattern=None`
- pass through `max_features`, `ngram_range`, `min_df`, `max_df`
- BoW: `binary`; TF-IDF: `norm='l2'`

Return `(X, vectorizer)`.

Expect about `(5114, 5000)`. BoW cells are counts (int); TF-IDF is
float. Low IDF = common across ham and spam. High IDF spam terms are
the ones a classifier would actually use — you do not build that
classifier in this module.

### 11–12. Word2Vec and FastText

Message vector = mean of token vectors.

- Word2Vec: skip OOV; all-OOV message → zero row; `min_count=2`
- FastText: subwords, so typos still get a vector; `min_count=1`

Return `(X, model)`, shape about `(5114, 100)`.

This is **not** GloVe. GloVe is pretrained on a huge corpus and
frozen. You train these two on 5114 SMS messages. Neighbours will be
spam-flavoured, not Wikipedia-flavoured.

---

## Track B — Parallel notebooks (local, gitignored)

Import the graded functions. Do not reimplement them in cells.

Suggested split:

1. `nb_01_explore.ipynb` — quoting bug, 5507 vs 5509 vs 5114, class
   imbalance, length by label, placeholder / phone / URL counts, task 0
   plots.
2. `nb_02_preprocess.ipynb` — before/after: clean → tokenizer comparison
   → stopword keep-list rescue → filter drops → lemma vs stem → spam
   bigram ratios → freq bars → spam vs ham clouds.
3. `nb_03_vectorize.ipynb` — BoW vs TF-IDF vs embeddings (OOV, zero
   rows, `most_similar`). Closing markdown: Track C below.

One notebook is fine if three feels theatrical. Structure matters more
than file count.

---

## Track C — Objectives with no school task

Explain these. Do not invent graded APIs.

| Objective | Enough to say |
|-----------|----------------|
| What is NLP? | Computation on language. Here: classify SMS. |
| spaCy vs NLTK | NLTK = toolkit you assemble. spaCy = pipeline (tok, POS, NER) with models. This course uses NLTK. spaCy is not in `requirements.txt`. |
| GloVe | Pretrained global co-occurrence vectors. Contrast with Word2Vec trained here. |
| NER | Label spans: PERSON, ORG, DATE, MONEY. Useful; not tasked. |
| Sentiment | Lexicon vs trained classifier. Spam ≠ negative sentiment. |
| Topic modeling | Unsupervised themes (e.g. LDA). Spam/ham is supervised labels. |
| Text classifier | Next step: Logistic Regression / NB on the matrices you built. Out of scope. |
| What's next | Transformers, contextual embeddings. After you can explain TF-IDF without sweating. |

---

## Checker traps (read twice)

- Load path and quoting: checkers assume cleaned `SMSSpamCollection`
  in CWD.
- Phone regex **before** the general number pass.
- Emoji remove = space.
- `keep_words` on stopwords.
- Placeholder regex is strict `^<[A-Za-z]+>$`.
- `10-bag_of_words.py` vs heading `10-bow.py`.
- Vectorizer: already-tokenized input, so `str.split` and no lowercase.
- Word2Vec silent OOV vs FastText always-a-vector.
- Plot titles, figsize, `tight_layout` where specified — image matchers
  do not grade your vibes.
- Line length 79. Docstrings on module and every function.

---

## Suggested week

| Block | Do |
|-------|----|
| 1 | Extract, load correctly, explore + clean + tokenize |
| 2 | Stopwords, filter, lemma/stem, n-grams, freq/cloud |
| 3 | BoW, TF-IDF, Word2Vec, FastText |
| 4 | Notebook recap + Track C explanations out loud, no Google |

When every bullet in a subdirectory README is checked, that code
already passed the school checker. This folder's README bullets are
not checked yet — that is the work.

---

## Out of scope

- Running commands for you in the terminal (you asked not to)
- Implementing the `.py` files in this pass
- Adding spaCy / GloVe downloads / new numbered tasks
- Committing notebooks
- Overwriting [README.md](README.md)
