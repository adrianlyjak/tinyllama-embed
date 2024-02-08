"""
First, optimize bits of quantization
"""

import logging
import sys
import time
from typing import List

import optuna
from optuna import Trial
from transformers import (
    TrainingArguments,
    IntervalStrategy,
    AutoTokenizer,
    LlamaModel,
    TrainerCallback,
)

from train import TinyEmbedTrainer, PeftConfig
from train import load_model_for_training, prepare_dataset, load_model
from experiment_config import (
    get_config,
    write_config,
    record_trial_params,
    record_trial_result,
)
from dataclasses import dataclass
import os

seed = 42


@dataclass
class HyperParams:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = (
        "q_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    )
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 2e-10
    batch_size: int = 4
    infonce_temp: float = 0.05


def run(params: HyperParams = HyperParams()):
    cfg = get_config()
    resume = False
    if resume:
        version = cfg.get("all_version") or 0
    else:
        last = cfg.get("all_version") or 0
        version = last + 1
        write_config({**cfg, "all_version": version})

    print(f"running experiment {version} with {params}")
    base_model, tokenizer = load_model(bits=4)

    model = load_model_for_training(
        base_model,
        PeftConfig(
            r=params.r,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            target_modules=params.target_modules,
        ),
    )
    dataset = prepare_dataset(tokenizer, seed)
    # Access the new train and test datasets
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    log_dir = f"./logs/logs_{version}.0"
    results_dir = f"./results_{version}.0"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_train_batch_size=params.batch_size,
        logging_dir=log_dir,
        log_level="info",
        logging_strategy=IntervalStrategy.STEPS,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=2000,
        num_train_epochs=1,
        seed=seed,
        adam_beta1=params.adam_beta1,
        adam_beta2=params.adam_beta2,
        adam_epsilon=params.adam_epsilon,
        report_to="tensorboard",
    )

    trainer = TinyEmbedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    run()
