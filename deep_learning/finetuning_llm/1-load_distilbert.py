#!/usr/bin/env python3
"""
Bring the tools: tokenizer and sequence
classification model to the lab, please!
"""
import transformers


def load_distilbert(
        model_name,
        num_classes,
        id2label,
        label2id
):
    """
    Hand out tokenizer and model.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    model = (
        transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id
        )
    )

    return tokenizer, model
