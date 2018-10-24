from io import open
import unicodedata

import torch

class Data(object):
    def __init__(self, file_path, train_ratio=0.97, min_length=5, max_length=20):
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.min_length = min_length
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()

        self.vocab = set(["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.word2count = {"<PAD>" : 0, "<SOS>" : 0, "<EOS>" : 0, "<UNK>" : 0}
        self.vocab_size = 4

        self.x_train = list()
        self.y_train = list()
        self.len_train = list()
        self.x_val = list()
        self.y_val = list()
        self.len_val = list()

        self.run()

    def get_lengths(self):
        self.len_train = [len(sentence) for sentence in self.x_train]
        self.len_val = [len(sentence) for sentence in self.x_val]

    def create_vocabulary(self):
        for data in [self.x_train, self.y_train]:
            for sentence in data:
                for word in sentence:
                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word2index[word] = self.vocab_size
                        self.index2word.append(word)
                        self.word2count[word] = 0
                        self.vocab_size += 1

                    self.word2count[word] += 1

    def replace_unk(self):
        for data in [self.x_val, self.y_val]:
            for i, sentence in enumerate(data):
                for j, word in enumerate(sentence):
                    if word not in self.vocab:
                        data[i][j] = "<UNK>"

    def replace_with_ind(self):
        for data in [self.x_train, self.y_train, self.x_val, self.y_val]:
            for i, sentence in enumerate(data):
                for j, word in enumerate(sentence):
                    data[i][j] = self.word2index[word]

    def list_of_tensors(self):
        for data in [self.x_train, self.x_val, self.y_train, self.y_val]:
            for i, sentence in enumerate(data):
                data[i] = torch.LongTensor(sentence)

    def run(self):
        data = []
        inp = []
        out = []
        with open(self.file_path) as f:
            lines = f.readlines()
            for line in lines:
                data += line.split() + ["<EOS>"]

            for i in range(0, len(data), self.max_length):
                inp += [data[i : i + self.max_length]]
                out += [data[i + 1 : i + self.max_length + 1]]

            if len(inp[-1]) < self.max_length:
                inp.pop()
                out.pop()

            n_data = len(inp)
            print('Read %d lines' % (n_data))
            train_len = int(self.train_ratio*n_data)

            self.x_train = inp[:train_len]
            self.y_train = out[:train_len]
            self.x_val = inp[train_len:]
            self.y_val = out[train_len:]

            self.get_lengths()
            self.create_vocabulary()
            self.replace_unk()

            self.replace_with_ind()
            self.list_of_tensors()
