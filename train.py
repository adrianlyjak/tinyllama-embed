from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Any, Tuple

import torch
from torch.nn import functional as F
import math

from transformers import (
    LlamaModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
)
from datasets import load_dataset, DatasetDict, Dataset

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType


def load_model(bits: int = 4) -> Tuple[LlamaModel, AutoTokenizer]:
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

        def tokenize(path: str) -> Dict[str, torch.Tensor]:
            token_dict = self.tokenizer(
                examples[path],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return {
                "input_ids_" + path: token_dict["input_ids"],
                "attention_mask_" + path: token_dict["attention_mask"],
            }

        return {
            **tokenize("query"),
            **tokenize("pos"),
            **tokenize("neg"),
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
        model: LlamaModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        callbacks: Optional[List[TrainerCallback]] = None,
        infonce_temp: float = 0.1,
        log_cosine_similarities: bool = False,
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
            callbacks=callbacks,
        )
        self.infonce_temp = infonce_temp
        self.log_cosine_similarities = log_cosine_similarities
        # self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        # self.tokenizer = tokenizer

    def _map_collate_fn(self, batch):
        """
        pads and truncates the batch for each of the three inputs
        and returns a single dict, with each key
        shape the the List[Dict[str, torch.Tensor]] into a Dict[str, torch.Tensor]
        """

        def get_field(field: str, length: Optional[int] = None):
            attention = [item["attention_mask_" + field] for item in batch]
            inputs = [item["input_ids_" + field] for item in batch]
            # probably something wrong at the dataset level... but munge things into the right shape here,
            # truncate the batch to the max length, but also 0 pad those that are less than the max length
            # for both input_ids and attention_mask
            max_len = length if length is not None else max([sum(x) for x in attention])
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

        length = 0
        for field in ["query", "pos", "neg"]:
            attention = [item["attention_mask_" + field] for item in batch]
            length = max([sum(x) for x in attention] + [length])

        return {
            **get_field("query", length),
            **get_field("pos", length),
            **get_field("neg", length),
        }

    def compute_loss(
        self, model: LlamaModel, inputs: Dict[str, torch.Tensor], return_outputs=False
    ):

        batch_size = inputs["input_ids_query"].shape[0]
        query = get_embeddings(
            model, inputs["input_ids_query"], inputs["attention_mask_query"]
        )
        all_embeddings = get_embeddings(
            model,
            torch.cat((inputs["input_ids_pos"], inputs["input_ids_neg"]), dim=0),
            torch.cat(
                (inputs["attention_mask_pos"], inputs["attention_mask_neg"]), dim=0
            ),
        )

        # Compute InfoNCE loss
        temperature = self.infonce_temp
        # query = [Batch, Embedding] * [Embedding, Batch * 2] = [Batch, Batch * 2]
        scores = torch.matmul(query, all_embeddings.transpose(0, 1)) / temperature
        labels = torch.arange(batch_size, device=query.device)

        # Use cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(scores, labels)
        # log cosine similarity
        if self.log_cosine_similarities:

            # Compute cosine similarity between query and positive/negative
            sim = torch.nn.functional.cosine_similarity(
                query.repeat_interleave(len(all_embeddings), dim=0),
                all_embeddings.repeat(batch_size, 1),
            ).reshape(-1, len(all_embeddings))
            pos = sum([sim[i][i] for i in range(len(sim))]).item() / len(sim)
            neg = sum(
                [
                    sum([sim[i][j] for j in range(len(sim)) if j != i]).item()
                    / (len(all_embeddings) - 1)
                    for i in range(len(sim))
                ]
            ) / len(sim)
            self.log({"pos": pos, "neg": neg})
        return loss


def get_embeddings_from_text(
    model: LlamaModel,
    tokenizer: AutoTokenizer,
    text: List[str],
) -> torch.Tensor:
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    embeddings = get_embeddings(
        model, tokenized["input_ids"], tokenized["attention_mask"]
    )
    return embeddings


def get_embeddings(
    model: LlamaModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
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
    # Normalize the embeddings to unit length
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings
