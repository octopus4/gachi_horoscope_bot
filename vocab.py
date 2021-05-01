import json
import re


class Tokenizer:
    def __init__(self):
        self.regex = r"([^a-zA-Zа-яА-Я0-9\.\,\!\?\;\:\&\-\"\'\` ])"

    def tokenize(self, sentence: str):
        cleaned = re.sub(self.regex, "", sentence)
        trimmed_spaces = re.sub(r"[ ]+", " ", cleaned)
        last_dot_index = trimmed_spaces.find('.')
        if last_dot_index > 0:
            return trimmed_spaces[:last_dot_index + 1]
        return trimmed_spaces


class Vocab:
    START_TOKEN = '<sos>'
    END_TOKEN = '<eos>'
    PADDING_TOKEN = '<pad>'

    @staticmethod
    def restore():
        path = 'vocab.json'
        with open(path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            return Vocab(json_data)

    def __init__(self, json_data):
        self.tokenizer = Tokenizer()
        self.idx2char = {}
        self.char2idx = {}
        char2idx = json_data['char2idx']
        ngramms = json_data['ngramms']
        max_ngramm = int(json_data['max_ngramm'])
        for char_key in char2idx:
            self.char2idx[char_key] = char2idx[char_key]
            self.idx2char[char2idx[char_key]] = char_key
        self.ngramms = {}
        for ngramm_size in ngramms:
            self.ngramms[int(ngramm_size)] = ngramms[ngramm_size]
        self.__max_ngramm = max_ngramm

    def tokenize(self, sequence):
        sequence = self.tokenizer.tokenize(sequence)
        tokens = []
        i, n = 0, len(sequence)
        while i < n:
            ngramm_added = False
            for k in range(self.__max_ngramm, 1, -1):
                if i < n - k + 1 and sequence[i:i + k] in self.ngramms[k]:
                    tokens.append(self.char2idx[sequence[i:i + k]])
                    ngramm_added = True
                    i += k
                    break
            if not ngramm_added:
                tokens.append(self.char2idx[sequence[i]])
                i += 1
        return tokens

    def detokenize(self, sequence):
        return ''.join([self.idx2char[idx] for idx in sequence.numpy()])

    def __len__(self):
        return len(self.char2idx)
