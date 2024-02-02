from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Any

import torch
from torch.nn import functional as F

from transformers import (
    LlamaModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from datasets import load_dataset, DatasetDict, Dataset

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType


def load_model(bits: int = 4) -> (LlamaModel, AutoTokenizer):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if bits == 4
        else BitsAndBytesConfig(load_in_8bit=True)
    )

    model = LlamaModel.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
        device_map="auto",
        quantization_config=bnb_config,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
        add_eos_token=True,
        use_fast=True,
    )
    # consider and experiment withadding a specific pad token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    return (model, tokenizer)


def prepare_dataset(tokenizer: AutoTokenizer, seed: int) -> DatasetDict:
    base_dataset = load_dataset(
        "andersonbcdefg/synthetic_tuples_gpt35_turbo", split="train"
    )

    dataset = base_dataset.map(
        Tokenize(tokenizer).tokenize_function,
        batched=True,
        cache_file_name="./cache/tokenized_datasets",
    )

    train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)

    return train_test_split


class Tokenize:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    # Tokenize the dataset
    def tokenize_function(self, examples: Dict[str, str]) -> Dict[str, torch.Tensor]:
        max_len = 512
        tokenized_query = self.tokenizer(
            examples["query"],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tokenized_pos = self.tokenizer(
            examples["pos"],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tokenized_neg = self.tokenizer(
            examples["neg"],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {
            "input_ids_query": tokenized_query["input_ids"],
            "attention_mask_query": tokenized_query["attention_mask"],
            "input_ids_pos": tokenized_pos["input_ids"],
            "attention_mask_pos": tokenized_pos["attention_mask"],
            "input_ids_neg": tokenized_neg["input_ids"],
            "attention_mask_neg": tokenized_neg["attention_mask"],
        }


@dataclass
class PeftConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]


def load_model_for_training(
    model: LlamaModel,
    peft_config: PeftConfig = PeftConfig(
        r=8,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ],
    ),
) -> LlamaModel:
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.target_modules,
        inference_mode=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)
    return model


class TinyEmbedTrainer(Trainer):
    """
    Subclasses hugging face trainer to do all the needed things:
    - embedding training needs multiple inputs to train on (query, positive, negative).
    -   collate function to reshape the dict and pad the batch
    """

    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        callback: Optional[Any] = None,
    ):
        # Consider reworking the model's signature to conform to training expectations
        args.remove_unused_columns = False
        super().__init__(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=self._map_collate_fn,
            callbacks=callback,
        )
        # self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        # self.tokenizer = tokenizer

    def _map_collate_fn(self, batch):
        """
        pads and truncates the batch for each of the three inputs
        and returns a single dict, with each key
        shape the the List[Dict[str, torch.Tensor]] into a Dict[str, torch.Tensor]

        """

        def get_field(field: str):
            attention = [item["attention_mask_" + field] for item in batch]
            inputs = [item["input_ids_" + field] for item in batch]
            # probably something wrong at the dataset level... but munge things into the right shape here,
            # truncate the batch to the max length, but also 0 pad those that are less than the max length
            # for both input_ids and attention_mask
            max_len = max([sum(x) for x in attention])
            attention_mask = []
            input_ids = []
            for i in range(len(attention)):
                attention_i = attention[i]
                input_ids_i = inputs[i]
                attention_trunc = attention_i[:max_len]
                input_ids_trunc = input_ids_i[:max_len]
                attention_pad = attention_trunc + [0] * (max_len - len(attention_trunc))
                input_ids_pad = input_ids_trunc + [self.tokenizer.pad_token_id] * (
                    max_len - len(input_ids_trunc)
                )
                attention_mask.append(attention_pad)
                input_ids.append(input_ids_pad)

            return {
                ("attention_mask_" + field): torch.tensor(attention_mask),
                ("input_ids_" + field): torch.tensor(input_ids),
            }

        return {**get_field("query"), **get_field("pos"), **get_field("neg")}

    def _get_embeddings(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden states
        # Compute the indices of the last non-padding tokens
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        embeddings = hidden_states[
            torch.arange(hidden_states.size(0)), last_token_indices, :
        ]
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def compute_loss(self, model, inputs, return_outputs=False):
        from datetime import datetime

        # print iso utc 8601 time with milliseconds
        query_embeddings = self._get_embeddings(
            model, inputs["input_ids_query"], inputs["attention_mask_query"]
        )
        pos_embeddings = self._get_embeddings(
            model, inputs["input_ids_pos"], inputs["attention_mask_pos"]
        )
        neg_embeddings = self._get_embeddings(
            model, inputs["input_ids_neg"], inputs["attention_mask_neg"]
        )

        # Compute InfoNCE loss
        pos_similarity = torch.sum(query_embeddings * pos_embeddings, dim=1)
        neg_similarity = torch.sum(query_embeddings * neg_embeddings, dim=1)

        # Compute InfoNCE loss
        pos_similarity = torch.sum(query_embeddings * pos_embeddings, dim=1)
        neg_similarity = torch.sum(query_embeddings * neg_embeddings, dim=1)
        losses = -torch.log(
            torch.exp(pos_similarity)
            / (torch.exp(pos_similarity) + torch.exp(neg_similarity))
        )
        return losses.mean()
