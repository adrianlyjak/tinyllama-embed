"""
First, optimize bits of quantization
"""

import logging
import sys

import optuna
from optuna import Trial
from transformers import TrainingArguments, IntervalStrategy

from train import TinyEmbedTrainer, PeftConfig
from train import load_model_for_training, prepare_dataset, load_model


seed = 42

"""
First, optimize bits of quantization
"""


def objective(trial: Trial):
    r = trial.suggest_int("r", 8, 8)
    alpha = trial.suggest_int("alpha", 64, 64)
    dropout = trial.suggest_float("dropout", 0.1, 0.1)
    modules = [
        "q_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ]
    batch_size = trial.suggest_categorical("batch_size", [4, 4])
    adam_beta1 = trial.suggest_float("adam_beta1", 0.9, 0.9)
    adam_beta2 = trial.suggest_float("adam_beta2", 0.999, 0.999)
    adam_epsilon = trial.suggest_float("adam_epsilon", 1e-8, 1e-8)
    bits = trial.suggest_categorical("bits", [8, 4])

    base_model, tokenizer = load_model(bits=bits)
    model = load_model_for_training(
        base_model,
        PeftConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=modules),
    )
    dataset = prepare_dataset(tokenizer, seed)
    # Access the new train and test datasets
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results",
        per_device_train_batch_size=batch_size,
        logging_dir="./logs",
        log_level="info",
        logging_strategy=IntervalStrategy.STEPS,
        gradient_checkpointing=True,
        logging_steps=5,
        max_steps=100,
        save_steps=1000,
        seed=seed,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        report_to="tensorboard",
    )

    trainer = TinyEmbedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=False, trial=trial)
    return result.training_loss


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def run():
    study = optuna.create_study(
        study_name="bits-quantization-5000",
        storage="sqlite:///experiments.db",
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=2)
