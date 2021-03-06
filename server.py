import os

from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
from loader import Loader

token = os.getenv('TELEGRAM_TOKEN')
print('retrieved token from env successfully' if token is not None else 'failed to retrieve token')
loader = Loader()
model, sampler = loader.load()
print('loaded model and sampler instances successfully')
updater = Updater(token)


def start(update: Update, callback_context: CallbackContext) -> None:
    user = update.message.from_user
    print(f'receiving start command from {user.username}')
    keyboard = [
        # fire
        [
            KeyboardButton("♈ Овен"),
            KeyboardButton("♌ Лев"),
            KeyboardButton("♐ Стрелец"),
        ],
        # earth
        [
            KeyboardButton("♉ Телец"),
            KeyboardButton("♍ Дева"),
            KeyboardButton("♑ Козерог"),
        ],
        # air
        [
            KeyboardButton("♊ Близнецы"),
            KeyboardButton("♎ Весы"),
            KeyboardButton("♒ Водолей"),
        ],
        # water
        [
            KeyboardButton("♋ Рак"),
            KeyboardButton("♏ Скорпион"),
            KeyboardButton("♓ Рыбы"),
        ]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard)
    update.message.reply_text('Hey buddy, you got the wrong door', reply_markup=reply_markup)


def handle_action(update: Update, callback_context: CallbackContext) -> None:
    user = update.message.from_user
    print(f'handling bot action from {user.username}')
    user_input = update.message.text
    sign = user_input[2:]
    response = sampler.sample_by_sign(sign)
    update.message.reply_text(response)


def main():
    print('started bot successfully')
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text, handle_action))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
