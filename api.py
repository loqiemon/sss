from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

app = FastAPI()

# Загрузка модели и токенизатора
model = DistilBertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = DistilBertTokenizer.from_pretrained('./trained_model')

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(message: Message):
    inputs = tokenizer(message.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    label_map = {0: 'низкий', 1: 'средний', 2: 'высокий'}
    return {"label": label_map[pred_label]}