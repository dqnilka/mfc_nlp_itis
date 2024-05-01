from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

TOKEN = ''
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands=['start'], state='*')
async def send_welcome(message: types.Message):
    user_first_name = message.from_user.first_name
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    await message.reply(f"Привет, {user_first_name}! Я Мария МФЦ - бот. Нажмите /register для регистрации.",
                        reply_markup=keyboard)


if __name__ == '__main__':
    executor.start_polling(dp)
