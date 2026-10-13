#!/usr/bin/env python3
"""
Trina the Trainer runs school for the Scribe.
"""
import transformers


def train_model(
        model,
        training_args,
        train_dataset,
        eval_dataset,
        test_dataset,
        tokenizer,
        data_collator,
        compute_metrics,
        model_save_name
):
    """
    Train, test, and save the scribe
    and its quill.
    """
    trina = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    train_results = trina.train()
    test_results = trina.evaluate(
        eval_dataset=test_dataset
    )
    trina.save_model(output_dir=model_save_name)
    tokenizer.save_pretrained(
        save_directory=model_save_name
    )

    return trina, train_results, test_results
