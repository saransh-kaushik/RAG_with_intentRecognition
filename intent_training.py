import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np


os.environ["WANDB_MODE"] = "disabled"



# Sample training data(change it as your requirements). Here I am using an example of a arestaurant menu
# 0: Menu Inquiry, 1: Order Request, 2: Pricing Inquiry
data = {
    "text": [
        "What is on the menu?", "Do you have drinks?", "I want a medium pizza with sausage",
        "How much is a small pizza?", "Can I get a large salad?", "What's the price of the cheese bread?",
        "I’d like to order a sandwich with turkey and cheddar", "What are the toppings available?",
        "How much does a soda can cost?", "Tell me about the desserts", "Add a coke to my order","can you get me a medium pizza with pepperoni and shrimps as toppings?",
        "can you add a large coke with that?"
    ],
    "label": [
        0, 0, 1, 2, 1, 2, 1, 0, 2, 0, 1, 1, 1
    ]
}



# Load data into a Dataset
dataset = Dataset.from_dict(data)
train_test = dataset.train_test_split(test_size=0.2)
train_data, test_data = train_test['train'], train_test['test']

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Tokenize data
def preprocess_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_data = train_data.map(preprocess_data, batched=True)
test_data = test_data.map(preprocess_data, batched=True)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train model
trainer.train()

# Save the trained model
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")

