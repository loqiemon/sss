import gradio as gr
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Загрузка модели и токенизатора
model = DistilBertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = DistilBertTokenizer.from_pretrained('./trained_model')

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    label_map = {0: 'низкий', 1: 'средний', 2: 'высокий'}
    return label_map[pred_label]

interface = gr.Interface(fn=predict, inputs="text", outputs="text")
interface.launch()