from peft import LoraConfig, TaskType
from transformers import (
    LlamaModel,
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    IntervalStrategy,
)
from datasets import load_dataset
import torch
from torch.nn import functional as F


class TinyEmbedTrainer(Trainer):
    def __init__(self, model, args, train_dataset, tokenizer):
        super().__init__(model, args)
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs)
        raise NotImplementedError("I am but a poor boy, from a poor family. Scalamoose")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden states

        # Compute the indices of the last non-padding tokens
        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1

        # Retrieve the embeddings for each example in the batch
        query_embeddings = hidden_states[
            torch.arange(hidden_states.size(0)), last_token_indices, :
        ]

        # Assuming that pos and neg inputs are also passed in the same manner
        # Replace 'pos_input' and 'neg_input' with the actual input names
        pos_outputs = model(**pos_input, output_hidden_states=True)
        neg_outputs = model(**neg_input, output_hidden_states=True)

        pos_hidden_states = pos_outputs.hidden_states[-1]
        neg_hidden_states = neg_outputs.hidden_states[-1]

        pos_embeddings = pos_hidden_states[
            torch.arange(pos_hidden_states.size(0)), last_token_indices, :
        ]
        neg_embeddings = neg_hidden_states[
            torch.arange(neg_hidden_states.size(0)), last_token_indices, :
        ]

        # Normalize the embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        pos_embeddings = F.normalize(pos_embeddings, p=2, dim=1)
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=1)

        # Compute InfoNCE loss
        pos_similarity = torch.sum(query_embeddings * pos_embeddings, dim=1)
        neg_similarity = torch.sum(query_embeddings * neg_embeddings, dim=1)
        losses = -torch.log(
            torch.exp(pos_similarity)
            / (torch.exp(pos_similarity) + torch.exp(neg_similarity))
        )
        return (losses.mean(), outputs) if return_outputs else losses.mean()


def run():
    seed = 42
    model = LlamaModel.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
    )

    # consider and experiment withadding a specific pad token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["query"],
            examples["pos"],
            examples["neg"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    dataset = load_dataset(
        "andersonbcdefg/synthetic_tuples_gpt35_turbo", split="train"
    ).map(tokenize_function, batched=True)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)

    # Access the new train and test datasets
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=32,
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
        inference_mode=False,
    )

    model.add_adapter(peft_config)

    trainer = TinyEmbedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=50,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    run()
