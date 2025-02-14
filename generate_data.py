from transformers import pipeline
import pandas as pd

# Генерация текстовых сообщений
generator = pipeline('text-generation', model='gpt2')

messages = []
labels = []

for _ in range(100):
    prompt = "Сообщение о происшествии: "
    message = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    label = "низкий"  # или "средний", "высокий" в зависимости от логики
    messages.append(message)
    labels.append(label)

# Сохранение данных в CSV
data = {'message': messages, 'label': labels}
df = pd.DataFrame(data)
df.to_csv('incident_data.csv', index=False)