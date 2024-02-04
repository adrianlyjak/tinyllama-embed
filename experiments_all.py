"""
First, optimize bits of quantization
"""

import logging
import sys
import time
import math

import optuna
from optuna import Trial
from transformers import (
    TrainingArguments,
    IntervalStrategy,
    AutoTokenizer,
    LlamaModel,
    TrainerCallback,
)
from typing import List
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
        """
        lora dropout = 0
        bits = 4
        reduce  range of adam epsilons to explore
        reduce range of adame beta to explore
        reduce range of adam beta2 to explore
        lora alpha [64,128,256,512]
        infonce max of 0.0625, explore lower
        r = 8
        batch size max 8
        """
        r_exp = trial.suggest_int("r_exp", 3, 3)
        r = 2**r_exp
        trial.set_user_attr("r", r)
        modules = [
            "q_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]
        adam_beta1 = trial.suggest_float("adam_beta1", 0.89, 0.91, step=0.005)
        adam_beta2 = trial.suggest_float("adam_beta2", 0.990, 0.999, step=0.001)
        adam_epsilon = trial.suggest_float("adam_epsilon", 1e-11, 1e-9, log=True)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.0, step=0.1)
        infonce_temp_exp_high = 8
        infonce_temp_exp = trial.suggest_int("infonce_temp_exp", 1, 6, step=1)
        infonce_temp = 2**infonce_temp_exp / (2**infonce_temp_exp_high)
        trial.set_user_attr("infonce_temp", infonce_temp)

        # For lora_alpha, suggest a power of 2
        lora_alpha_exp = trial.suggest_int("lora_alpha_exp", 6, 9, step=1)
        lora_alpha = 2**lora_alpha_exp
        trial.set_user_attr("lora_alpha", lora_alpha)
        batch_size_exp = trial.suggest_int("batch_size_exp", 0, 4, step=1)
        batch_size = 2**batch_size_exp
        trial.set_user_attr("batch_size", batch_size)
        bits = trial.suggest_categorical("bits", [4])

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
                "batch_size": batch_size,
                "bits": bits,
                "infonce_temp": infonce_temp,
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
            infonce_temp=infonce_temp,
        )

        start_time = time.time()
        result = trainer.train(resume_from_checkpoint=False, trial=trial)
        end_time = time.time()
        duration_seconds = end_time - start_time
        loss_history = [
            log["loss"] for log in trainer.state.log_history if "loss" in log
        ]
        initial_loss = loss_history[0]
        static_initial_loss = 0.7
        most_recent_loss = smooth(loss_history, 0.4)[-3:]
        mean_final_loss = sum(most_recent_loss) / len(most_recent_loss)
        change_in_loss_per_second = (
            static_initial_loss - mean_final_loss
        ) / duration_seconds
        # write experiment
        record_trial_result(
            "all",
            self.version,
            run_number,
            {
                "loss": mean_final_loss,
                "initial_loss": initial_loss,
                "final_loss": loss_history[-1],
                "change_in_loss_per_second": change_in_loss_per_second,
                "history": loss_history,
            },
        )
        trial.set_user_attr("loss", mean_final_loss)
        trial.set_user_attr("final_loss", loss_history[-1])
        trial.set_user_attr("duration_seconds", duration_seconds)
        trial.set_user_attr("initial_loss", initial_loss)
        trial.set_user_attr("change_in_loss_per_second", change_in_loss_per_second)
        trial.set_user_attr("version_trial", f"{self.version}.{run_number}")

        return change_in_loss_per_second


def smooth(
    scalars: List[float], weight: float
) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


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
