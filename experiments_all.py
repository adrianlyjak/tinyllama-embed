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
    TrainerState,
    TrainerCallback,
)
from typing import Callable, List
from train import TinyEmbedTrainer, PeftConfig
from train import load_model_for_training, prepare_dataset, load_model
from experiment_config import get_config, write_config
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
    max_steps: int
    smoothing: float
    last_samples: int
    time_objective: bool

    def objective(self, trial: Trial):

        r_exp = trial.suggest_int("r_exp", 4, 4)
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
        adam_beta1 = trial.suggest_float("adam_beta1", 0.9, 0.9, step=0.005)
        adam_beta2 = trial.suggest_float("adam_beta2", 0.999, 0.999, step=0.001)
        adam_epsilon = trial.suggest_float("adam_epsilon", 2e-10, 2e-10, log=True)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.0, step=0.1)
        infonce_temp = trial.suggest_float("infonce_temp", 0.025, 0.025, step=0.005)

        trial.set_user_attr("infonce_temp", infonce_temp)

        # For lora_alpha, suggest a power of 2
        lora_alpha_scale_exp = trial.suggest_int("lora_alpha_scale_exp", 1, 1, step=1)
        lora_alpha = r * 2**lora_alpha_scale_exp
        trial.set_user_attr("lora_alpha", lora_alpha)
        batch_size_exp = trial.suggest_int("batch_size_exp", 2, 2, step=1)
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

        log_dir = f"./logs/logs_{self.version}.{trial.number}"
        results_dir = f"./results"
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
            logging_steps=1,
            max_steps=self.max_steps,
            save_steps=100000,
            seed=seed,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            report_to="tensorboard",
        )

        start_time = time.time()

        def get_loss() -> LossInfo:
            current_time = time.time()
            duration_seconds = current_time - start_time
            loss_history = [
                log["loss"] for log in trainer.state.log_history if "loss" in log
            ]

            static_initial_loss = 0.7
            most_recent_loss = smooth(loss_history, self.smoothing)[
                -1 * self.last_samples :
            ]
            mean_loss = sum(most_recent_loss) / len(most_recent_loss)
            change_in_loss_per_second = (
                static_initial_loss - mean_loss
            ) / duration_seconds
            return LossInfo(
                mean_smoothed_recent_loss=mean_loss,
                loss_history=loss_history,
                change_in_loss_per_second=change_in_loss_per_second,
                seconds=duration_seconds,
                target=change_in_loss_per_second if self.time_objective else mean_loss,
            )

        trainer = TinyEmbedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            infonce_temp=infonce_temp,
            callbacks=[
                # OptunaPruningCallback(
                #     trial,
                #     lambda: get_loss().target,
                # )
            ],
        )

        result = trainer.train(resume_from_checkpoint=False, trial=trial)
        loss = get_loss()

        trial.set_user_attr("loss", loss.mean_smoothed_recent_loss)
        trial.set_user_attr("final_loss", loss.loss_history[-1])
        trial.set_user_attr("duration_seconds", loss.seconds)
        trial.set_user_attr("initial_loss", loss.loss_history[0])
        trial.set_user_attr("change_in_loss_per_second", loss.change_in_loss_per_second)

        return loss.target


@dataclass
class LossInfo:
    mean_smoothed_recent_loss: float
    loss_history: List[float]
    change_in_loss_per_second: float
    seconds: float
    target: float


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, evaluate: Callable[[], float]):
        self.trial = trial
        self.evaluate = evaluate

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        # Retrieve the objective metric from the current evaluation.
        if len(state.log_history):
            objective = self.evaluate()

            # Report the current objective to Optuna and potentially prune the trial.
            self.trial.report(objective, step=state.global_step)
            should_prune = self.trial.should_prune()
            if should_prune:
                message = "Trial was pruned at step {}.".format(state.global_step)
                raise optuna.exceptions.TrialPruned(message)


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
    resume = True
    if resume:
        version = cfg.get("all_version") or 0
    else:
        last = cfg.get("all_version") or 0
        version = last + 1
        write_config({**cfg, "all_version": version})

    base8, tokenizer = load_model(bits=8)
    base4, _ = load_model(bits=4)
    is_time_objective = False
    inst = TrainExperiment(
        version=version,
        base8=base8,
        base4=base4,
        tokenizer=tokenizer,
        max_steps=3000,
        smoothing=0.99,
        last_samples=20,
        time_objective=is_time_objective,
    )
    study = optuna.create_study(
        study_name=f"all-{version}",
        storage="sqlite:///experiments.db",
        load_if_exists=resume,
        direction="maximize" if is_time_objective else "minimize",
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=400, n_startup_trials=3),
    )
    study.optimize(inst.objective, n_trials=1)


if __name__ == "__main__":
    run()
