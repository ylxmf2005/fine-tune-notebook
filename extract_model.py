from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BloomForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
import math

name1 = "text"
name2 = "labels"

data = load_dataset("json", data_files="sec_pretrain.sectoday.json", 
                split={'train': 'train[:5%]', 'validation': 'train[95%:]'})

model_checkpoint = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast = True)

# 从 JSON 导入 dataset 默认全是 train

def transform_features(batch):
    transformed_batch = {"text": [], "labels": []}
    for text, meta in zip(batch["text"], batch["meta"]): # 只能遍历一级 key
        transformed_batch["text"].append(text)
        transformed_batch["labels"].append(meta["comment"]) # 二级 key 另取
    return transformed_batch

def filter_texts(dataset, p1 = 5, p2 = 15, min_length = None, max_length = None):
    texts = dataset[name1]
    labels = dataset[name2]
    lengths = [len(tokenizer.encode(text)) for text in texts]
    sorted_lengths = sorted(lengths)
    l = len(sorted_lengths)
    lower_index = int(l * p1 / 100)
    upper_index = int(l - l * p2 /100)
    if (min_length == None): min_length = sorted_lengths[lower_index]
    if (max_length == None): max_length = sorted_lengths[upper_index]
    valid_indices = [i for i, text in enumerate(texts) if min_length <= len(tokenizer.encode(text)) <= max_length]
    filtered_texts = [texts[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    return Dataset.from_dict({name1: filtered_texts, name2: filtered_labels})

def remove_none(example):
    return example["labels"] is not None

data["train"] = data["train"].map(transform_features, batched = True, num_proc = 4,
                                  remove_columns = ["text", "meta", "src", "_id"]) 
# map 类似新加，重合则覆盖；remove_columns 是在参数传入 map 之后，map 进行新加之前删掉对应键值对
data["validation"] = data["validation"].map(transform_features, batched = True, num_proc = 4,
                                            remove_columns = ["text", "meta", "src", "_id"])

data["train"] = data["train"].filter(remove_none)
data["validation"] = data["validation"].filter(remove_none)

data["train"] = filter_texts(data["train"])
data["validation"] = filter_texts(data["validation"])

def get_maxlen(texts):
    text_lengths = [len(tokenizer.encode(text)) for text in texts]
    return max(text_lengths)
text_lengths = [len(tokenizer.encode(text)) for text in data["train"]["text"]]
MAX_LENGTH = max(get_maxlen(data["train"]["text"]), get_maxlen(data["validation"]["text"]))

"""
sns.set()
plt.figure(figsize = (10, 6))
sns.histplot(text_lengths, kde = True, bins = "auto")
plt.show()

lengths, counts = np.unique(text_lengths, return_counts=True)
distribution = dict(zip(lengths, counts))
lengths_to_texts = {length: [] for length in lengths}
for length, text in zip(text_lengths, data["train"]["text"]):
    lengths_to_texts[length].append(text)
num_items = int(len(distribution) * 0.8)
subset_distribution = dict(list(distribution.items())[:num_items])
tmp = list(subset_distribution.items())[100:1000]
for i in tmp:
    length, count = i
    print(f"Length: {length}, Count: {count}")
    for text in lengths_to_texts[length]:
        print(f"Text: {text}")
print(subset_distribution)
"""

# 对于所有数据按 MAX_LENGTH 填充太浪费资源，考虑不使用 padding="max_length"（也不要加 return_tensors="pt"），后面使用 data_collator 在批处理训练时进行动态填充

"""

def tokenize_examples(examples):
    tokenized_text = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)
    tokenized_labels = tokenizer(examples["labels"], truncation=True, max_length=MAX_LENGTH)
    
    examples["input_ids"] = tokenized_text.input_ids
    examples["attention_mask"] = tokenized_text.attention_mask
    examples["labels"] = tokenized_labels.input_ids

    return examples
"""

def tokenize_examples(examples):
    tokenized_text = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)
    tokenized_labels = tokenizer(examples["labels"], truncation=True, max_length=MAX_LENGTH)
    
    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "labels": tokenized_labels["input_ids"],
    }


"""
"[UNK]" 对应 id 为 0
"[CLS]" 对应 id 为 1
"[SEP]" 对应 id 为 2
"[PAD]" 对应 id 为 3
"[MASK]" 对应 id 为 4。
"""


data["train"] = data["train"].map(tokenize_examples, batched=True, num_proc=4,
                                  remove_columns = ["text", "labels"])
data["validation"] = data["validation"].map(tokenize_examples, batched=True, num_proc=4,
                                            remove_columns = ["text", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# print(data["train"][0])

def save_as_jsonl(dataset, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in dataset:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

save_as_jsonl(data["train"], "train_data.jsonl")
save_as_jsonl(data["validation"], "validation_data.jsonl")

model = BloomForCausalLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
# training_args = TrainingArguments(model_name)

training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=3,              
    per_device_train_batch_size=4,  
    learning_rate=2e-5, 
    warmup_steps=500,               
    weight_decay=0.01,             
    logging_dir='./logs',           
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    data_collator = data_collator,
)

# 批处理训练时使用 data_collator 进行动态填充
trainer.train()
# eval_results = trainer.evaluate()
# print(eval_results)
