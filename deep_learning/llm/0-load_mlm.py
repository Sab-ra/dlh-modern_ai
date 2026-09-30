#!/usr/bin/env python3
"""
Load a pretrained RoBERTa model for
fill-in-the-blank.
"""
import transformers


def load_mlm(model_name):
    """
    Load a RoBERTa model that is ready for guesses
    Args:
        model_name(str): name of the pretrained model
    Returns:
        RobertaForMaskedLM: model set for business
    """
    model = (
        transformers.RobertaForMaskedLM.from_pretrained(
            model_name
        )
    )
    model.eval()

    return model
