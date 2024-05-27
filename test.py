import requests  # Импортируем библиотеку requests для выполнения HTTP-запросов
from dotenv import load_dotenv  # Импортируем функцию load_dotenv для загрузки переменных окружения из файла .env
import os  # Импортируем модуль os для работы с переменными окружения
from src.constants import label2text  # Импортируем словарь label2text из модуля src.constants

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем URL сервера из переменной окружения
URL = os.getenv("URL")

# Функция для тестирования здоровья эндпоинта /saiga
def test_health_saiga():
    text = ""  # Пустой текст для тестового запроса
    response = requests.post(f"{URL}/saiga", json={"text": text})  # Выполняем POST-запрос к эндпоинту /saiga
    assert response.status_code == 200  # Проверяем, что статус-код ответа 200 (OK)

# Функция для тестирования здоровья эндпоинта /classify
def test_health_bert():
    text = ""  # Пустой текст для тестового запроса
    response = requests.post(f"{URL}/classify", json={"text": text})  # Выполняем POST-запрос к эндпоинту /classify
    assert response.status_code == 200  # Проверяем, что статус-код ответа 200 (OK)

# Функция для тестирования правильности классификации текста
def test_label():
    text = "Какая единовременная компенсационная выплата при рождении сына?"  # Тестовый текст для классификации
    response = requests.post(f"{URL}/classify", json={"text": text})  # Выполняем POST-запрос к эндпоинту /classify
    # Проверяем, что предсказанная метка соответствует ожидаемому тексту
    assert label2text[response.json()["label"]] == "Детская карта"