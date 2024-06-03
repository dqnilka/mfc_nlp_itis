import logging
import os
from datetime import datetime

import telebot
import time
import random
import requests  # Импорт библиотеки requests для выполнения HTTP-запросов

from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import psycopg2
from telebot import types

from config import questions_dict
from src.utils import answer_with_label
# Импорт библиотеки для работы с PostgreSQL
import psycopg2

load_dotenv()

# Определение переменных окружения
PORT = "8080"
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
URL = os.getenv("URL")
token = os.getenv("API_TOKEN")
operator_id = os.getenv("OPERATOR_ID")

bot = telebot.TeleBot(token)

# Замените на реальный ID оператора
OPERATOR_ID = operator_id
bot.user_question = None
bot.user_answer = None

help_msg = (
    "*Добро пожаловать в чат-бот поддержки!*\n\n"
    "Вот что я могу для вас сделать:\n\n"
    "1. *Задать вопрос*: Просто напишите свой вопрос, и я постараюсь ответить как можно точнее.\n"
    "2. *Оценить ответ*: После получения ответа, вы можете оценить его, нажав на кнопку с оценкой от 1 до 5.\n"
    "3. *Связаться с оператором*: Если ответ вас не устроил, и вы оценили его на 1, я предложу вам отправить вопрос оператору.\n\n"
    "*Спасибо, что пользуетесь нашими услугами! 🙂*"
)

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(filename="bot.log", encoding="utf-8", level=logging.DEBUG)


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(f"New message: {message.text}")
    if message.from_user.id == OPERATOR_ID:
        send_operator_welcome(message)
    else:
        send_user_welcome(message)


def generate_random_number(length=30):
    digits = "0123456789"
    random_number = ''.join(random.choices(digits, k=length))
    return random_number


def save_rating_to_db(user_name, rating, user_message, output_message):
    try:
        # Подключение к базе данных PostgreSQL
        conn = psycopg2.connect(
            host=db_host, database=db_name, user=db_user, password=db_password
        )
        cur = conn.cursor()
        logger.info(f"New message: {user_message}")

        # Генерация случайного числа
        random_id = generate_random_number()

        # Проверка типа данных перед кодированием
        user_message = "test"
        output_message = "test2"
        # Вставка данных в таблицу statistics
        cur.execute(
            "INSERT INTO statistics (id, user_name, rating, message, output_message, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (int(random_id), str(user_name), int(rating), str(user_message), str(output_message), datetime.now())
        )

        # Фиксация изменений и закрытие подключения
        conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Ошибка при сохранении оценки в базу данных:", error)


def send_user_welcome(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("❓ Задать вопрос")
    btn2 = types.KeyboardButton("📖 Помощь")
    markup.add(btn1, btn2)

    bot.send_message(message.chat.id,
                     "Привет, я МФЦ бот «Мария».\n"
                     "Я подскажу тебе ответ на любой твой вопрос касающийся работы многофункционального центра.",
                     reply_markup=markup)


def send_operator_welcome(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("📊 Просмотреть статистику")
    btn2 = types.KeyboardButton("❓ Просмотреть вопросы")
    markup.add(btn1, btn2)

    bot.send_message(message.chat.id,
                     "Привет, я МФЦ бот «Мария».\n"
                     "С моей помощью ты сможешь помогать пользователям получить информацию на свои вопросы!",
                     reply_markup=markup)


# Обработчик кнопки "❓ Задать вопрос"
@bot.message_handler(func=lambda message: message.text == "❓ Задать вопрос")
def ask_question(message):
    logger.info(f"New message: {message.text}")
    if message.from_user.id == OPERATOR_ID:
        bot.send_message(message.chat.id, "Эта команда недоступна для оператора.")
    else:
        hide_markup = types.ReplyKeyboardRemove()
        bot.send_message(message.chat.id, "Для получения ответа, напишите вопрос в чат с ботом.",
                         reply_markup=hide_markup)
        bot.register_next_step_handler(message, process_question)


def send_text_streaming(message):
    logger.info(f"New message: {message.text}")
    try:
        # Путь к файлу с ответом
        file_path = "answer.txt"

        # Открываем файл и читаем его
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Форматируем текст с использованием Markdown
        formatted_text = f"*Результат: \n*_{text}_"

        time.sleep(2)
        # Отправляем отформатированное сообщение
        bot.send_message(message.chat.id, formatted_text, parse_mode='Markdown')

        # После отправки ответа, предложить пользователю оценить работу
        send_rating_request(message)

    except Exception as e:
        # Если произошла ошибка, отправляем сообщение об ошибке
        bot.send_message(message.chat.id, f"Произошла ошибка: {e}")


def simulate_loading(chat_id, text, delay=1, iterations=1):
    message = bot.send_message(chat_id, text, parse_mode='Markdown')
    for i in range(iterations):
        time.sleep(delay)
        dots = '.' * (i % 3 + 1)
        bot.edit_message_text(text + dots, chat_id, message.message_id, parse_mode='Markdown')


def process_question(message):
    logger.info(f"User's question: {message.text}")
    question = message.text
    # Сохранение вопроса пользователя в контексте
    bot.answer_question = question
    # Здесь вы можете обработать вопрос пользователя
    simulate_loading(message.chat.id, f"ℹ️ Ваш вопрос принят: \n «_{question}_».\n"
                                      f"Начинаю формирование ответа, ожидайте")

    # Запрос на классификацию текста
    res_class = requests.post(f"{URL}/classify", json={"text": message.text})

    if res_class.status_code == 200:
        label = res_class.json()["label"]
        if label != 111:
            # Запрос на генерацию ответа
            # res_saiga = requests.post(f"{URL}/saiga", json={"text": message.text})

            # if res_saiga.status_code == 200:
            #    text = res_saiga.json()["prediction"]
            text = "Тест - Ответ"
            t = answer_with_label(text, label)
            file_path = "answer.txt"

            bot.user_answer = t
            bot.user_question = message.text

            # Открываем файл и читаем его
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(t)
            send_text_streaming(message)
    #         else:
    #             logger.error(res_saiga.text)
    #             bot.reply_to(message, "Произошла ошибка при обработке запроса.")
    #     else:
    #         bot.reply_to(
    #             message,
    #             "Кажется, я не совсем понял вопрос или он не относится к теме МФЦ. Не могли бы вы его уточнить?",
    #         )
    # else:
    #     logger.error(res_class.text)
    #     bot.reply_to(message, "Произошла ошибка при обработке запроса.")


def send_rating_request(message):
    logger.info(f"New message: {message.text}")
    # Создание кнопок для оценки в одну строку
    markup = types.InlineKeyboardMarkup(row_width=5)
    buttons = [types.InlineKeyboardButton(str(i), callback_data=str(i)) for i in range(1, 6)]
    markup.add(*buttons)

    # Отправка сообщения с призывом к оценке
    bot.send_message(message.chat.id, "Пожалуйста, оцените работу чат-бота по 5-ти бальной шкале.", reply_markup=markup)


def show_main_buttons(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("❓ Задать вопрос")
    btn2 = types.KeyboardButton("📖 Помощь")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, "Желаете узнать что-то еще?", reply_markup=markup)


def show_operator_buttons(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("📊 Просмотреть статистику")
    btn2 = types.KeyboardButton("❓ Просмотреть вопросы")
    btn_back = types.KeyboardButton("◀️ Вернуться назад")
    markup.add(btn1, btn2, btn_back)
    bot.send_message(message.chat.id, "Желаете узнать что-то еще?", reply_markup=markup)


# Функция для отображения категорий для просмотра статистики
def show_statistics_categories(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    categories = ["Стаститка: Детская карта", "Стаститка: Паспорт", "Стаститка: ЗАГС", "Стаститка: Строительство",
                  "Стаститка: Выплаты", "Стаститка: Налоги", "Стаститка: Общая статистика", "Стаститка: Лицензии"]
    buttons = [types.KeyboardButton(category) for category in categories]
    btn_back = types.KeyboardButton("◀️ Вернуться назад")
    markup.add(*buttons, btn_back)
    bot.send_message(message.chat.id, "Укажите категорию для просмотра статистики:", reply_markup=markup)


# Функция для отображения категорий для просмотра вопросов
def show_questions_categories(message):
    logger.info(f"New message: {message.text}")
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    categories = ["Детская карта", "Паспорт", "ЗАГС", "Строительство", "Выплаты", "Налоги", "Лицензии"]
    buttons = [types.KeyboardButton(category) for category in categories]
    btn_back = types.KeyboardButton("◀️ Вернуться назад")
    markup.add(*buttons, btn_back)
    bot.send_message(message.chat.id, "Укажите категорию для просмотра вопросов:", reply_markup=markup)


# Обработчик кнопки "Просмотреть статистику" (для оператора)
@bot.message_handler(
    func=lambda message: message.text == "📊 Просмотреть статистику" and message.from_user.id == OPERATOR_ID)
def view_statistics(message):
    logger.info(f"New message: {message.text}")
    show_statistics_categories(message)


# Обработчик кнопки "Просмотреть вопросы" (для оператора)
@bot.message_handler(
    func=lambda message: message.text == "❓ Просмотреть вопросы" and message.from_user.id == OPERATOR_ID)
def view_questions(message):
    logger.info(f"New message: {message.text}")
    show_questions_categories(message)


# Обработчик выбора категории для вопросов
@bot.message_handler(
    func=lambda message: message.text in ["Детская карта", "Паспорт", "ЗАГС", "Строительство", "Выплаты", "Налоги",
                                          "Лицензии"])
def handle_category_q_selection(message):
    logger.info(f"New message: {message.text}")
    category = message.text
    if category in questions_dict:
        markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        questions = questions_dict[category]  # Получение списка вопросов для выбранной категории
        buttons = [types.KeyboardButton(question) for question in questions]
        btn_back = types.KeyboardButton("◀️ Вернуться назад")
        markup.add(*buttons, btn_back)
        bot.send_message(message.chat.id, "Выберите вопрос:", reply_markup=markup)


# Обработчик выбора вопроса для ответа
@bot.message_handler(func=lambda message: message.text in sum(questions_dict.values(), []))
def handle_question_selection(message):
    logger.info(f"New message: {message.text}")
    time.sleep(1)
    question = message.text
    bot.selected_question = question
    time.sleep(1)
    bot.send_message(message.chat.id, f"*Вопрос:* {question}\n\n"
                                      "Напишите ответ в текущий чат, чтобы автоматически ответить пользователю.",
                     parse_mode='Markdown')
    bot.register_next_step_handler(message, handle_operator_response)


def handle_operator_response(message):
    logger.info(f"New message: {message.text}")
    response = message.text
    time.sleep(2)
    bot.send_message(message.chat.id, "✅ Ваш ответ был успешно отправлен пользователю и добавлен в базу знаний.")
    # Здесь можно добавить код для отправки ответа пользователю и сохранения в базу знаний.


# Обработчик выбора категории для статистики
@bot.message_handler(
    func=lambda message: message.text in ["Стаститка: Детская карта", "Стаститка: Паспорт", "Стаститка: ЗАГС",
                                          "Стаститка: Строительство", "Стаститка: Выплаты", "Стаститка: Налоги",
                                          "Стаститка: Общая статистика", "Стаститка: Лицензии"])
def handle_category_selection(message):
    logger.info(f"New message: {message.text}")
    category = message.text
    if message.from_user.id == OPERATOR_ID:
        if category in ["Стаститка: Детская карта", "Стаститка: Паспорт", "Стаститка: ЗАГС", "Стаститка: Строительство",
                        "Стаститка: Выплаты", "Стаститка: Налоги", "Стаститка: Общая статистика",
                        "Стаститка: Лицензии"]:
            # Показать статистику
            num_questions = random.randint(20, 100)
            satisfaction_level = round(random.uniform(2.0, 5.0), 1)
            response = (f"Свод статистики за текущий месяц: *{category}*\n"
                        f"Кол-во обработанных вопросов: *{num_questions}*\n"
                        f"Средний уровень удовлетворенности: *{satisfaction_level} балла*")
            time.sleep(1)
            bot.send_message(message.chat.id, response, parse_mode='Markdown')
    else:
        # Обработка для пользователя (например, если он нажал на категории в помощи)
        bot.send_message(message.chat.id, "Эта команда недоступна для пользователя.")


# Обработчик кнопки "📖 Помощь"
@bot.message_handler(func=lambda message: message.text == "📖 Помощь")
def help_message(message):
    logger.info(f"New message: {message.text}")
    if message.from_user.id == OPERATOR_ID:
        bot.send_message(message.chat.id, "Эта команда недоступна для оператора.")
    else:
        bot.send_message(message.chat.id,
                         "Вот список команд, которые я понимаю:\n"
                         "/start - начать\n"
                         "/help - помощь")


# Обработчик команды /help
@bot.message_handler(commands=['help'])
def help_command(message):
    logger.info(f"New message: {message.text}")
    bot.send_message(message.chat.id,
                     help_msg, parse_mode='Markdown')


# Обработчик кнопки "Новый вопрос"
@bot.message_handler(func=lambda message: message.text == "🔄 Новый вопрос")
def new_question(message):
    logger.info(f"New message: {message.text}")
    hide_markup = types.ReplyKeyboardRemove()
    bot.send_message(message.chat.id, "Пожалуйста, задайте свой новый вопрос:", reply_markup=hide_markup)
    bot.register_next_step_handler(message, process_question)


# Обработчик кнопки "Назад"
@bot.message_handler(func=lambda message: message.text == "◀️ Вернуться назад")
def go_back(message):
    logger.info(f"New message: {message.text}")
    if message.from_user.id == OPERATOR_ID:
        send_operator_welcome(message)
    else:
        send_user_welcome(message)


# Обработчик кнопок оценки
@bot.callback_query_handler(func=lambda call: call.data in ['1', '2', '3', '4', '5'])
def callback_rating(call):
    rating = call.data
    # Удаление сообщения с кнопками оценки
    bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
    logger.info(f"New message: user make a rate")
    username = (
        call.from_user.username
        if call.from_user.username
        else call.from_user.first_name
    )
    user_message = bot.user_question
    output_message = bot.user_answer

    print(username)
    print(rating)
    print(user_message)
    print(output_message)
    if rating == '1':
        # Создание кнопки для отправки вопроса оператору
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("Направить вопрос оператору", callback_data="send_to_operator"))
        bot.send_message(call.message.chat.id, "Спасибо за оценку!", reply_markup=markup)

        save_rating_to_db(username, rating, user_message, output_message)
    else:
        save_rating_to_db(username, rating, user_message, output_message)
        bot.send_message(call.message.chat.id, "Спасибо за оценку!")
        if call.from_user.id == OPERATOR_ID:
            show_operator_buttons(call.message)
            time.sleep(1)
        else:
            show_main_buttons(call.message)
            time.sleep(1)


# Обработчик кнопки "Направить вопрос оператору"
@bot.callback_query_handler(func=lambda call: call.data == "send_to_operator")
def send_to_operator(call):
    question = bot.answer_question  # Получение сохраненного вопроса пользователя
    time.sleep(2)
    bot.send_message(call.message.chat.id, f"Ваш вопрос: «_{question}_», был успешно отправлен оператору МФЦ. "
                                           f"Ответ будет направлен в текущий чат после того, как будет сформулирован.",
                     parse_mode='Markdown')
    show_main_buttons(call.message)


# Запуск бота
bot.polling()
