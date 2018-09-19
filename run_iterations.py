import time
import random

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from nltk import bleu_score

from helper import Helper

class Run_Iterations(object):
    def __init__(self, model, train_in_seq, train_out_seq, index2word, batch_size,
                 num_iters, learning_rate, tracking_seed=False, val_in_seq=[],
                 val_out_seq=[], print_every=1, plot_every=1):
        self.use_cuda = torch.cuda.is_available()
        self.model = model
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.tracking_seed = tracking_seed
        self.print_every = print_every
        self.plot_every = plot_every

        self.index2word = index2word
        ''' Lists that will contain data in the form of tensors. '''
        # Training data.
        self.train_in_seq = train_in_seq
        self.train_out_seq = train_out_seq
        self.train_samples = len(self.train_in_seq)

        # Validation data.
        self.val_in_seq = val_in_seq
        self.val_out_seq = val_out_seq
        self.val_samples = len(self.val_in_seq)

        self.help_fn = Helper()

    def train_iters(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every self.print_every
        plot_loss_total = 0  # Reset every self.plot_every

        lm_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.lm.parameters()))
        lm_optimizer = optim.RMSprop(lm_trainable_parameters, lr=self.learning_rate)

        print('Beginning Model Training.')

        for epoch in range(1, self.num_iters + 1):
            for i in range(0, self.train_samples, self.batch_size):
                input_variables = self.train_in_seq[i : i + self.batch_size] # Batch Size x Sequence Length
                target_variables = self.train_in_seq[i : i + self.batch_size] # Batch Size x Sequence Length

                loss = self.model.train(input_variables, target_variables, self.criterion, lm_optimizer)
                print_loss_total += loss
                plot_loss_total += loss

                # now = time.time()
                # print('Completed %.4f Percent of Epoch %d in %s Minutes' % ((i + self.batch_size)/ self.train_samples * 100,
                #                                                             epoch, self.help_fn.as_minutes(now - start)))

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.help_fn.time_slice(start, epoch / self.num_iters),
                                             epoch, epoch / self.num_iters * 100, print_loss_avg))

            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # self.help_fn.show_plot(plot_losses)

    def evaluate_specific(self, in_seq, out_seq, seed_length):
        input = [self.index2word[j] for j in in_seq]
        output = [self.index2word[j] for j in out_seq]
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

        print('BLEU Score', bleu_score.corpus_bleu([output_sentence], [output]))
        print('-----------------------------------------------------------------')

    def evaluate_randomly(self, n=10):
        for i in range(n):
            ind = random.randrange(self.val_samples)
            for seed_length in range(len(self.val_in_seq[ind])//2):
                self.evaluate_specific(self.val_in_seq[ind], self.val_out_seq[ind], seed_length)

            print('\n')
