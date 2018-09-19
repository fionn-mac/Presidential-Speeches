from io import open
import unicodedata

import torch

class Data(object):
    def __init__(self, file_path, train_ratio=0.8, max_length=10):
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()

        self.vocab = set(["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.word2count = {"<PAD>" : 0, "<SOS>" : 0, "<EOS>" : 0, "<UNK>" : 0}
        self.vocab_size = 4

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()

        self.run()

    def create_vocabulary(self):
        for sentence in self.x_train:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab.add(word)
                    self.word2index[word] = self.vocab_size
                    self.index2word.append(word)
                    self.word2count[word] = 0
                    self.vocab_size += 1

                self.word2count[word] += 1

    def replace_unk(self):
        for i, sentence in enumerate(self.x_val):
            for j, word in enumerate(sentence):
                if word not in self.vocab:
                    self.x_val[i][j] = "<UNK>"

    def replace_with_ind(self):
        for data in [self.x_train, self.y_train, self.x_val, self.y_val]:
            for i, sentence in enumerate(data):
                for j, word in enumerate(sentence):
                    data[i][j] = self.word2index[word]

    def list_of_tensors(self):
        for data in [self.x_train, self.x_val, self.y_train, self.y_val]:
            for i, sentence in enumerate(data):
                data[i] = torch.LongTensor(sentence)
                if self.use_cuda: data[i].cuda()

    def run(self):
        data = []
        with open(self.file_path) as f:
            lines = f.readlines()

            for line in lines:
                data.append(["<SOS>"] + line.split())

            n_data = len(data)
            print('Read %d lines' % (n_data))

            self.x_train = data[:int(self.train_ratio*n_data)]
            self.x_val = data[int(self.train_ratio*n_data):]

            self.x_train = sorted(self.x_train, key=len)
            self.x_val = sorted(self.x_val, key=len)

            self.create_vocabulary()
            self.replace_unk()

            for inp, out in zip([self.x_train, self.x_val], [self.y_train, self.y_val]):
                for sentence in inp:
                    out.append(sentence[1:] + ["<EOS>"])

            self.replace_with_ind()
            self.list_of_tensors()
