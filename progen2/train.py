

import os
import time
import random
import argparse

import torch

from tokenizers import Tokenizer
from models.progen.configuration_progen import ProGenConfig
from models.progen.modeling_progen import ProGenForCausalLM
from datasets import load_dataset, DatasetDict

from transformers import Trainer, TrainingArguments

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

tokenizer = create_tokenizer_custom(file='tokenizer_.json')

tokenizer.enable_padding(pad_token="<|pad|>")

out = tokenizer.encode("ABCDEFG")

# print(out)

# print(out.ids)
# print(out.type_ids)
# print(out.tokens)
# print(out.offsets)
# print(out.attention_mask)

outs = tokenizer.encode_batch(["ABCD*", "ABCDEFG"])

# print(outs[0].ids)
# print(outs[0].tokens)
# print(outs[1].ids)
# print(outs[0].attention_mask)
# print(outs[1].attention_mask)

# print("tokenizer vocab size: {}, tokenizer padding information: {}, tokenizer model: {}".format(tokenizer.get_vocab_size(), tokenizer.padding, tokenizer.model))

# tokenizer_model=tokenizer.model


# print(tokenizer)

# 分词方法
# def tokenize(example):
#     output = tokenizer.encode(example["sequence"])
#     return {"ids": output.ids, "type_ids": output.type_ids, "tokens": output.tokens, "attention_mask": output.attention_mask}
def tokenize(example):
    output = tokenizer.encode(example["sequence"])
    return {"input_ids": output.ids}

def tokenize_batch(example):
    output = tokenizer.encode_batch(example["sequence"])
    # print(output[0])
    ids = []
    type_ids = []
    tokens = []
    attention_masks = []
    for item in output:
        ids.append(item.ids)
        type_ids.append(item.type_ids)
        tokens.append(item.tokens)
        attention_masks.append(item.attention_mask)
    return {"ids": ids, "type_ids": type_ids, "tokens": tokens, "attention_mask": attention_masks}
    # return {"ids": output.ids}
trainset = load_dataset("csv", data_files=["data/sequences.csv"])
print(trainset)
# print(trainset["train"][0])

# 使用non-batch方法加载数据
tokenized_dataset = trainset.map(tokenize, remove_columns=trainset["train"].column_names)
# 使用batch方法加载数据
tokenized_bateched_dataset = trainset.map(tokenize_batch, batched=True, remove_columns=trainset["train"].column_names)

print(tokenized_dataset)
print(tokenized_bateched_dataset)

# 创建config用于配置模型
config = ProGenConfig(vocab_size=tokenizer.get_vocab_size(), n_positions=512, n_ctx=1024, n_embd=512, n_layer=6, n_head=8)

# 根据模型创建模型
model = ProGenForCausalLM(config)
# print(model)

model_size = sum(t.numel() for t in model.parameters())
print(f"Progen size: {model_size/1000**2:.1f}M parameters")

# 训练代码

args = TrainingArguments(
    output_dir="checkpoints",
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

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([tokenized_dataset["train"][i] for i in range(5)])

for key in out:
    print(f"{key} shape: {out[key].shape}")