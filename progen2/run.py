from datasets import load_dataset
from data_utils import *
from transformers import Trainer, TrainingArguments
from models.progen.configuration_progen import ProGenConfig
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import PreTrainedTokenizerFast
trainset = load_dataset("csv", data_files=["data/sequences.csv"])


# 训练使用
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
# 推理使用
# tokenizer = create_tokenizer_custom("tokenizer.json")

# out = tokenizer.encode("ABCD")
# print(tokenizer("ABCD"))
# print(out)

def tokenize(example):
    # out = tokenizer.encode(example['sequence'])

    # return {"input_ids": out.ids, "type_ids": out.type_ids, "tokens": out.tokens, "attention_mask": out.attention_mask} 
    out = tokenizer(example['sequence'])
    return {"input_ids": out['input_ids'], "token_type_ids": out["token_type_ids"], "attention_mask": out["attention_mask"]}

tokenized_dataset = trainset.map(
    tokenize,
    batched=False,
    remove_columns=trainset["train"].column_names
)
print(tokenized_dataset)

block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

config = ProGenConfig(vocab_size=32, n_positions=128, n_embd=512, n_layer=6, n_head=8)

# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': "[PAD]"})

model = ProGenForCausalLM(config)


lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# print(lm_datasets)
args = TrainingArguments(
    output_dir="checkpoints/",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["train"]
)

trainer.train()
