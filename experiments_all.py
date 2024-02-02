"""
First, optimize bits of quantization
"""

import logging
import sys
import time

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

"""
First, optimize bits of quantization
"""


@dataclass()
class TrainExperiment:
    version: str
    base8: LlamaModel
    base4: LlamaModel
    tokenizer: AutoTokenizer

    def objective(self, trial: Trial):

        r = trial.suggest_categorical("r", [1, 2, 4, 8, 16, 32])
        modules = [
            "q_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]
        adam_beta1 = trial.suggest_float("adam_beta1", 0.85, 0.95, step=0.01)
        adam_beta2 = trial.suggest_float("adam_beta2", 0.99, 0.9999, step=0.0001)
        adam_epsilon = trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.5, step=0.1)

        # For lora_alpha, suggest a power of 2
        lora_alpha = trial.suggest_categorical("lora_alpha", [2, 4, 8, 16, 32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
        bits = trial.suggest_categorical("bits", [4, 8])

        base_model = self.base4 if bits == 4 else self.base8
        tokenizer = self.tokenizer

        model = load_model_for_training(
            base_model,
            PeftConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=modules,
            ),
        )
        dataset = prepare_dataset(tokenizer, seed)
        # Access the new train and test datasets
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # write experiment
        run_number = record_trial_params(
            "all",
            self.version,
            {
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_epsilon": adam_epsilon,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "r": r,
            },
        )

        log_dir = f"./logs_{self.version}.{run_number}"
        results_dir = f"./results_{self.version}.{run_number}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=results_dir,
            per_device_train_batch_size=batch_size,
            logging_dir=log_dir,
            log_level="info",
            logging_strategy=IntervalStrategy.STEPS,
            gradient_checkpointing=True,
            logging_steps=5,
            max_steps=150,
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

        start_time = time.time()
        result = trainer.train(resume_from_checkpoint=False, trial=trial)
        end_time = time.time()
        duration_seconds = end_time - start_time
        loss_history = [
            log["loss"] for log in trainer.state.log_history if "loss" in log
        ]
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        change_in_loss_per_second = (initial_loss - final_loss) / duration_seconds
        # write experiment
        record_trial_result(
            "all",
            self.version,
            run_number,
            {
                "loss": final_loss,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "change_in_loss_per_second": change_in_loss_per_second,
                "history": loss_history,
            },
        )
        trial.set_user_attr("final_loss", final_loss)
        trial.set_user_attr("duration_seconds", duration_seconds)
        trial.set_user_attr("initial_loss", initial_loss)
        trial.set_user_attr("change_in_loss_per_second", change_in_loss_per_second)

        return change_in_loss_per_second


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def run():
    cfg = get_config()
    resume = False
    if resume:
        version = cfg.get("all_version") or 0
    else:
        last = cfg.get("all_version") or 0
        version = last + 1
        write_config({**cfg, "all_version": version})

    base8, tokenizer = load_model(bits=8)
    base4, _ = load_model(bits=4)
    inst = TrainExperiment(
        version=version, base8=base8, base4=base4, tokenizer=tokenizer
    )
    study = optuna.create_study(
        study_name=f"all-{version}",
        storage="sqlite:///experiments.db",
        load_if_exists=resume,
        direction="maximize",
    )
    study.optimize(inst.objective, n_trials=100)


if __name__ == "__main__":
    run()
