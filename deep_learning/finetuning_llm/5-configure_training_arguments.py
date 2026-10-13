#!/usr/bin/env python3
"""
Write the school rulebook for training the scribe.
"""
import transformers


def configure_training_args(
        output_dir,
        epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        learning_rate,
        weight_decay,
        metric_for_best_model,
        seed
):
    """
    Return Training Arguments for fine-tuning.
    """
    transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        push_to_hub=True,
        seed=seed
    )
