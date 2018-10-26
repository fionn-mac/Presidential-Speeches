from io import open
import unicodedata
import operator

import torch

class Data(object):
    def __init__(self, file_path, train_ratio=0.97, min_length=5, max_length=20):
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.min_length = min_length
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()

        self.vocab = ["<PAD>", "<UNK>"]
        self.word2index = {}
        self.index2word = []
        self.word2count = {}
        self.vocab_size = 2

        self.x_train = list()
        self.y_train = list()
        self.len_train = list()
        self.x_val = list()
        self.y_val = list()
        self.len_val = list()

        self.run()

    def create_vocabulary(self):
        for sentence in self.x_train:
            for word in sentence:
                if word not in self.word2count:
                    self.word2count[word] = 0
                self.word2count[word] += 1

        sorted_x = sorted(self.word2count.items(), key=operator.itemgetter(1), reverse=True)

        self.vocab += [tup[0] for tup in sorted_x[:25000]]
        self.vocab_size = len(self.vocab)
        del sorted_x

        for i, word in enumerate(self.vocab):
            self.word2index[word] = i
            self.index2word.append(word)

        self.vocab = set(self.vocab)

    def replace_unk(self):
        in_data_lists = []
        out_data_lists = []

        for inp in [self.x_train, self.x_val]:
            clean_in = []
            clean_out = []

            for i, sentence in enumerate(inp):
                unk_count = 0
                for j, word in enumerate(sentence):
                    if word not in self.vocab:
                        inp[i][j] = "<UNK>"
                        unk_count += 1

                if unk_count < j // 2:
                    clean_in.append(inp[i][:-1])
                    clean_out.append(inp[i][1:])

            in_data_lists.append(clean_in)
            out_data_lists.append(clean_out)

        self.x_train = in_data_lists[0]
        self.x_val = in_data_lists[1]
        del in_data_lists

        self.y_train = out_data_lists[0]
        self.y_val = out_data_lists[1]
        del out_data_lists

    def get_lengths(self):
        self.len_train = [len(sentence) for sentence in self.x_train]
        self.len_val = [len(sentence) for sentence in self.x_val]

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

        with open(self.file_path) as f:
            lines = f.readlines()
            for line in lines:
                data += line.split() + ["<EOS>"]

        for i in range(0, len(data), self.max_length):
            inp += [data[i : i + self.max_length + 1]]
        del data

        if len(inp[-1]) < self.max_length + 1:
            inp.pop()

        n_data = len(inp)
        print('Read %d lines' % (n_data))
        train_len = int(self.train_ratio*n_data)

        self.x_train = inp[:train_len]
        self.x_val = inp[train_len:]
        del inp

        print('Building Vocabulary.')
        self.create_vocabulary()

        print('Replacing words outside vocabulary with <UNK>.')
        self.replace_unk()

        self.get_lengths()

        print('Converting to PyTorch Tensors.')
        self.replace_with_ind()
        self.list_of_tensors()
