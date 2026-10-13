#!/usr/bin/env python3
"""
Turn the phrases in the chest (dataset)
into number sequences (tokens).
"""


def tokenize_and_map(
        dataset,
        tokenizer,
        max_length,
        truncation,
        batched
):
    """
    Tokenize train, validation, and test
    data.
    """

    def grab_handful(examples):
        """
        Helper: Rewrite one handful of
        phrases as tokens.
        """
        return (
            tokenizer(
                examples['text'],
                truncation=truncation,
                max_length=max_length
            )
        )

    tokenized = dataset.map(grab_handful, batched=batched)

    tok_train   = tokenized['train']
    tok_val     = tokenized['validation']
    tok_test    = tokenized['test']

    return tok_train, tok_val, tok_test
