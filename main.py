from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import torch
from PIL import Image
from torchvision import transforms
from alfabet import number_to_word
import numpy as np

import os
#TOKEN = os.environ["TOKEN"]
TOKEN = 1305871705:AAETizfOeuLot3R_mSx3kjClr0JA1SzHQzI
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Hello!\nSend me a picture!")

@dp.message_handler(content_types = types.ContentTypes.PHOTO)
async def process_kek_command(message: types.Message):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    await message.photo[-1].download("photo_{}.jpg".format(message.chat.id))
    input_image = Image.open("photo_{}.jpg".format(message.chat.id))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    output = ""
    with torch.no_grad():
        output = model(input_batch)
        output = number_to_word[np.argmax(output[0]).item()]
    await message.reply(output)



@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("When you send an image, you will receive information about the objects located on it, and for an another type of input data, I will start repeating you")


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)


if __name__ == '__main__':
    executor.start_polling(dp)
