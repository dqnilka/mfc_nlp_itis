import asyncio  # Импорт модуля asyncio для асинхронного программирования
from telebot.async_telebot import AsyncTeleBot  # Импорт асинхронного бота для Telegram
from telebot import types  # Импорт типов для работы с Telegram API
from dotenv import load_dotenv  # Импорт функции для загрузки переменных окружения из файла .env
import os  # Импорт модуля os для работы с переменными окружения
import requests  # Импорт библиотеки requests для выполнения HTTP-запросов
from src.utils import answer_with_label  # Импорт утилиты для обработки ответов
import logging  # Импорт модуля для логирования
import psycopg2  # Импорт библиотеки для работы с PostgreSQL

# Загрузка переменных окружения из файла .env
load_dotenv()

# Определение переменных окружения
PORT = "8080"
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
URL = os.getenv("URL")
token = os.getenv("API_TOKEN")

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(filename="bot.log", encoding="utf-8", level=logging.DEBUG)

# Инициализация асинхронного бота с токеном
bot = AsyncTeleBot(token)

# Создание клавиатуры с кнопками для оценки
kb = types.InlineKeyboardMarkup(
    [
        [
            types.InlineKeyboardButton(text="1", callback_data="btn_types_1"),
            types.InlineKeyboardButton(text="2", callback_data="btn_types_2"),
            types.InlineKeyboardButton(text="3", callback_data="btn_types_3"),
            types.InlineKeyboardButton(text="4", callback_data="btn_types_4"),
            types.InlineKeyboardButton(text="5", callback_data="btn_types_5"),
        ]
    ]
)


# Функция для сохранения оценки в базу данных
def save_rating_to_db(user_name, rating, user_message, output_message):
    try:
        # Подключение к базе данных PostgreSQL
        conn = psycopg2.connect(
            host=db_host, database=db_name, user=db_user, password=db_password
        )
        cur = conn.cursor()

        # Вставка данных в таблицу statistics
        cur.execute(
            "INSERT INTO statistics (user_name, rating, message, output_message) VALUES (%s, %s, %s,  %s)",
            (user_name, rating, user_message, output_message),
        )

        # Фиксация изменений и закрытие подключения
        conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Ошибка при сохранении оценки в базу данных:", error)


# Обработчик команды /start
@bot.message_handler(commands=["start"])
async def send_welcome(message):
    await bot.reply_to(
        message,
        """\
Привет, я МФЦ бот.
Я подскажу тебе ответ на любой твой вопрос касающийся работы МФЦ\
""",
    )


# Обработчик всех текстовых сообщений
@bot.message_handler(func=lambda message: True)
async def message(message):
    logger.info(f"New message: {message.text}")

    # Запрос на классификацию текста
    res_class = requests.post(f"{URL}/classify", json={"text": message.text})

    if res_class.status_code == 200:
        label = res_class.json()["label"]

        if label != 111:
            # Запрос на генерацию ответа
            res_saiga = requests.post(f"{URL}/saiga", json={"text": message.text})

            if res_saiga.status_code == 200:
                text = res_saiga.json()["prediction"]
                await bot.reply_to(
                    message, answer_with_label(text, label), reply_markup=kb
                )
            else:
                logger.error(res_saiga.text)
                await bot.reply_to(message, "Произошла ошибка при обработке запроса.")
        else:
            await bot.reply_to(
                message,
                "Кажется, я не совсем понял вопрос или он не относится к теме МФЦ. Не могли бы вы его уточнить?",
            )
    else:
        logger.error(res_class.text)
        await bot.reply_to(message, "Произошла ошибка при обработке запроса.")


# Обработчик нажатий на кнопки оценки
@bot.callback_query_handler(func=lambda call: True)
async def callback_worker(call):
    rating = call.data.split("_")[2]
    username = (
        call.from_user.username
        if call.from_user.username
        else call.from_user.first_name
    )
    user_message = call.message.json["reply_to_message"]["text"]
    output_message = call.message.text
    logger.info(f"{username} rated {user_message} as {rating}")
    save_rating_to_db(username, rating, user_message, output_message)
    await bot.send_message(call.from_user.id, "Спасибо за вашу оценку!")


# Запуск бота
asyncio.run(bot.polling())