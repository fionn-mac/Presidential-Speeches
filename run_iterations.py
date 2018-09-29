import time
import random

from os import system

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from helper import Helper

class Run_Iterations(object):
    def __init__(self, model, train_in_seq, train_len, train_out_seq, word2index, index2word,
                 batch_size, num_iters, learning_rate, tracking_seed=None,
                 val_in_seq=[], val_len=[], val_out_seq=[], fold_size=500000, print_every=1, plot_every=1):
        self.use_cuda = torch.cuda.is_available()
        self.model = model
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.print_every = print_every
        self.plot_every = plot_every

        self.index2word = index2word
        self.word2index = word2index
        ''' Lists that will contain data in the form of tensors. '''
        # Training data.
        self.train_in_seq = train_in_seq
        self.train_len = train_len
        self.train_out_seq = train_out_seq
        self.train_samples = len(self.train_in_seq)
        self.fold_size = self.train_samples
        if fold_size: self.fold_size = fold_size

        # Validation data.
        self.val_in_seq = val_in_seq
        self.val_len = val_len
        self.val_out_seq = val_out_seq
        self.val_samples = len(self.val_in_seq)

        if tracking_seed:
            indexed_seed = []
            # Assuming tokens to be space separated, so no need for fancy tokenization.
            for word in tracking_seed.lower().split():
                if word in self.word2index: indexed_seed.append(self.word2index[word])
                else: indexed_seed.append(self.word2index["<UNK>"])

            self.tracking_seed = torch.LongTensor(indexed_seed).view(1, -1)
            if self.use_cuda: self.tracking_seed = self.tracking_seed.cuda()

        else:
            self.tracking_seed = None

        self.help_fn = Helper()

    def train_iters(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every self.print_every
        plot_loss_total = 0  # Reset every self.plot_every

        lm_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.lm.parameters()))

        in_folds = []
        out_folds = []
        len_folds = []
        for i in range(0, self.train_samples, self.fold_size):
            in_folds.append(self.train_in_seq[i : i + self.fold_size])
            out_folds.append(self.train_out_seq[i : i + self.fold_size])
            len_folds.append(self.train_len[i : i + self.fold_size])

        self.train_in_seq = in_folds
        self.train_out_seq = out_folds
        self.train_len = len_folds
        del in_folds, out_folds, len_folds

        lm_hidden = self.model.lm.init_hidden(self.batch_size)

        print('Beginning Model Training.')

        fold_number = 1
        for in_fold, out_fold, len_fold in zip(self.train_in_seq, self.train_out_seq, self.train_len):
            # Initialize optimizer
            learning_rate = self.learning_rate
            lm_optimizer = optim.Adam(lm_trainable_parameters, lr=learning_rate)

            # Convert fold contents to cuda
            if self.use_cuda:
                in_fold = self.help_fn.to_cuda(in_fold)
                out_fold = self.help_fn.to_cuda(out_fold)

            fold_size = len(in_fold)
            fraction = fold_size // 10

            print('Starting Fold  :', fold_number)

            for epoch in range(1, self.num_iters + 1):
                for i in range(0, fold_size, self.batch_size):
                    input_variables = in_fold[i : i + self.batch_size] # Batch Size x Sequence Length
                    target_variables = out_fold[i : i + self.batch_size] # Batch Size x Sequence Length
                    input_lengths = len_fold[i : i + self.batch_size]

                    if len(input_variables) != self.batch_size:
                        continue

                    loss, lm_hidden = self.model.train(input_variables, input_lengths, target_variables,
                                                       lm_hidden, self.criterion, lm_optimizer)
                    print_loss_total += loss
                    plot_loss_total += loss

                    if i > 0 and (i - self.batch_size) // fraction < i // fraction:
                        now = time.time()
                        print('Completed %.2f Percent of Epoch %d in %s Minutes' % ((i + self.batch_size) / fold_size * 100,
                                                                                    epoch, self.help_fn.as_minutes(now - start)))

                        if self.tracking_seed is not None:
                            self.evaluate_specific(self.tracking_seed, self.tracking_seed, self.tracking_seed.size()[0])

                if epoch % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (self.help_fn.time_slice(start, epoch / self.num_iters),
                                                 epoch, epoch / self.num_iters * 100, print_loss_avg))

                if epoch % self.plot_every == 0:
                    plot_loss_avg = plot_loss_total / self.plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                if epoch % 2 == 0:
                    learning_rate /= 2
                    lm_optimizer = optim.Adam(lm_trainable_parameters, lr=learning_rate)

            # Convert fold contents back to cpu
            if self.use_cuda:
                in_fold = self.help_fn.to_cpu(in_fold)
                out_fold = self.help_fn.to_cpu(out_fold)

            self.help_fn.show_plot(plot_losses)
            fold_number += 1
            print('\n')

    def evaluate_specific(self, in_seq, out_seq, seed_length):
        input = [self.index2word[j] for j in in_seq[0]]
        output = [self.index2word[j] for j in out_seq[0]]
        print('>', input)
        print('~', seed_length)

        output_words = self.model.evaluate(in_seq, seed_length)
        try:
            target_index = output_words[0].index("<EOS>") + 1
        except ValueError:
            target_index = len(output_words[0])

        output_words = output_words[0][:target_index]

        output_sentence = ' '.join(output_words)
        print('<', output_sentence)

        print('-----------------------------------------------------------------')

    def evaluate_randomly(self, n=10):
        if self.use_cuda:
            self.val_in_seq = self.help_fn.to_cuda(self.val_in_seq)
            self.val_out_seq = self.help_fn.to_cuda(self.val_out_seq)

        for i in range(n):
            ind = random.randrange(self.val_samples)
            for seed_length in range(1, len(self.val_in_seq[ind]) // 2, 3):
                # Get output for given seed
                self.evaluate_specific(self.val_in_seq[ind].view(1, -1),
                                       self.val_out_seq[ind].view(1, -1),
                                       seed_length)

            print('\n')
