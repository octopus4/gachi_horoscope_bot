import datetime
import numpy as np
import torch

from torch.autograd import Variable
from model import LanguageModel
from vocab import Vocab

months = [
    'января',
    'февраля',
    'марта',
    'апреля',
    'мая',
    'июня',
    'июля',
    'августа',
    'сентября',
    'октября',
    'ноября',
    'декабря',
]

signs = {
    "Овен",
    "Лев",
    "Стрелец",
    "Телец",
    "Дева",
    "Козерог",
    "Близнецы",
    "Весы",
    "Водолей",
    "Рак",
    "Скорпион",
    "Рыбы"
}


class Sampler:
    def __init__(self, model: LanguageModel, device: torch.cuda.Device):
        self.__model = model
        self.__device = device
        self.__vocab = Vocab.restore()

    def sample_by_sign(self, sign):
        if sign not in signs:
            return ''
        today = datetime.datetime.now()
        d, m = today.day, months[today.month - 1]
        seed = f"{sign}, {d} {m}: "
        continuation = self.__sample__(192, seed, 0.4)
        return seed + continuation

    def __is_eos__(self, input_encoded):
        return input_encoded[0] == self.__vocab.char2idx[Vocab.END_TOKEN]

    def __sample__(self, max_len=128, seed="", temperature=1.0):
        vocab = self.__vocab
        input_encoded = []
        for char in seed.lower():
            input_encoded.append(vocab.char2idx[char])

        result = ''
        hidden = self.__model.init_hidden(1).to(self.__device)
        input_var = Variable(torch.LongTensor([input_encoded]))
        with torch.no_grad():
            while not self.__is_eos__(input_encoded):
                output, hidden = self.__model(input_var.to(self.__device), hidden, temperature)
                a = output.cpu().data[0, -1, :].numpy()
                p = np.exp(a) / np.exp(a).sum()
                input_encoded = [np.random.choice(len(vocab), p=p)]
                if self.__is_eos__(input_encoded):
                    break
                result += vocab.idx2char[input_encoded[0]]
                if len(result) > max_len:
                    break
                input_var = Variable(torch.LongTensor(input_encoded)).view(1, 1)
        return result
