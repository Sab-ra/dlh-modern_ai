#!/usr/bin/env python3
"""
Continue a sentence with a writing model.
Probably this is the chit that you hate
as in-line suggestions! Got ya bastard!
"""
import transformers


def create_text_generator(
        model_name,
        prompt,
        max_new_tokens,
        temperature,
        repetition_penalty,
        no_repeat_ngram_size
):
    """
    Build a vriter and let it continue
    a prompt.
    Args:
        model_name(str): Which w(v)riter to invite.
        prompt (str): The words it must start from.
        max_new_tokens (int): How many new pieces
            it may add.
        temperature (float): How daring the text
            word choice may be.
        repetition_penalty (float): Push away from
            words already used.
        no_repeat_ngram_size (int): Block a repeated
            phrase of this many words.
    Returns:
        generator: The ready-made writer
        output: The continued text.
    """
    generator = transformers.pipeline(
        'text-generation',
        model=model_name
    )
    generator.tokenizer.pad_token_id = (
        generator.tokenizer.eos_token_id
    )
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    return generator, output
