from src.constants import label2text
import psycopg2
import os


def answer_with_label(text, label):
    ans = f"Категория вопроса: {label2text[label]} \nОтвет на вопрос: \n{text}"
    return ans
