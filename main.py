import pandas as pd
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
)
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import os
from peft import PeftModel, PeftConfig

# Класс для ведения диалога
class SaigaConversation:
    def __init__(self, message_template, system_prompt, start_token_id, bot_token_id):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        prompt = "".join(self.message_template.format(**msg) for msg in self.messages)
        prompt += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return prompt.strip()

# Модели данных для FastAPI
class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    prediction: str

class ClassificationInputData(BaseModel):
    text: str

class ClassificationOutputData(BaseModel):
    label: int

# Функция для среднеарифметического пуллинга
def average_pooling(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    masked_hidden_states = hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Функция для создания эмбеддингов
def create_embeddings(
    texts: List[str], model: AutoModel, tokenizer: AutoTokenizer, device: torch.device
) -> Tensor:
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded_inputs = tokenizer(
            list(batch_texts), padding=True, truncation=True, return_tensors="pt"
        )
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            model_output = model(**encoded_inputs)
        batch_embeddings = average_pooling(
            model_output.last_hidden_state, encoded_inputs["attention_mask"]
        )
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Функция для классификации текста
def classify_text(text):
    encoded_input = classification_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True
    ).to("cpu")
    with torch.no_grad():
        output = classification_model(**encoded_input)
    predicted_class_id = output.logits.argmax().item()
    return predicted_class_id

# Функция для генерации ответа
def generate_response(model, tokenizer, prompt, generation_config):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**input_ids, generation_config=generation_config, temperature=0.001)[0]
    output_ids = output_ids[len(input_ids["input_ids"][0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# Инициализация FastAPI приложения
app = FastAPI()

# Загрузка токенизатора и модели для создания эмбеддингов
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
embed_model = AutoModel.from_pretrained("finetuned_labse_model.pth").to("cuda:0")

# Загрузка данных и создание эмбеддингов для вопросов
df = pd.read_csv("data/row/train_dataset.csv", sep=",")
passage_embeddings = create_embeddings(
    df["QUESTION"].tolist(), embed_model, embed_tokenizer, "cuda:0"
)

# Инициализация FAISS индекса
faiss_index_file = "faiss/faiss_index.pkl"
if os.path.exists(faiss_index_file):
    faiss_index = faiss.read_index(faiss_index_file)
else:
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    faiss_index.add(passage_embeddings.cpu().numpy())
    faiss.write_index(faiss_index, faiss_index_file)

# Настройка модели Saiga
SAIGA_MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
SAIGA_BASE_MODEL_PATH = "TheBloke/Llama-2-7B-fp16"
SAIGA_DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
SAIGA_DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

saiga_tokenizer = AutoTokenizer.from_pretrained(SAIGA_MODEL_NAME, use_fast=False)
saiga_config = PeftConfig.from_pretrained(SAIGA_MODEL_NAME)
saiga_model = AutoModelForCausalLM.from_pretrained(
    SAIGA_BASE_MODEL_PATH,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
saiga_model = PeftModel.from_pretrained(
    saiga_model, SAIGA_MODEL_NAME, torch_dtype=torch.float16
)
saiga_model.eval()
saiga_generation_config = GenerationConfig.from_pretrained(SAIGA_MODEL_NAME)

# Загрузка модели и токенизатора для классификации
model_name = "DeepPavlov/rubert-base-cased"
classification_model = AutoModelForSequenceClassification.from_pretrained(
    "train_bert/results/checkpoint-750"
).to("cpu")
classification_tokenizer = AutoTokenizer.from_pretrained(
    model_name, model_max_length=512
)

# Эндпоинт для генерации ответа с использованием Saiga
@app.post("/saiga", response_model=OutputData)
def generate_saiga_response(input_data: InputData):
    client_request = input_data.text

    # Создание эмбеддингов для запроса и поиск ближайших соседей в FAISS индексе
    query_text = f"query: {client_request}"
    query_embedding = create_embeddings(
        [query_text], embed_model, embed_tokenizer, "cuda:0"
    )
    scores, indices = faiss_index.search(query_embedding.cpu().numpy(), 4)
    top_result_idx = indices[0][0]

    # Создание запроса для Saiga
    prompt = f"У меня есть вопрос: {client_request}. Также у меня есть ответ: {df.iloc[top_result_idx]['ANSWER']}. На основе этого ответа коротко ответь на исходный вопрос."
    conversation = SaigaConversation(
        message_template=SAIGA_DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=SAIGA_DEFAULT_SYSTEM_PROMPT,
        start_token_id=1,
        bot_token_id=9225,
    )
    conversation.add_user_message(prompt)
    saiga_prompt = conversation.get_prompt(saiga_tokenizer)

    # Генерация ответа
    saiga_response = generate_response(
        saiga_model, saiga_tokenizer, saiga_prompt, saiga_generation_config
    )
    return OutputData(prediction=saiga_response)

# Эндпоинт для классификации текста
@app.post("/classify", response_model=ClassificationOutputData)
def classify_text_endpoint(input_data: ClassificationInputData):
    text = input_data.text
    predicted_class_id = classify_text(text)
    return ClassificationOutputData(label=predicted_class_id)