import random

import torch

class Train_Network(object):
    def __init__(self, lm, index2word, teacher_forcing_ratio=0.5):
        self.lm = lm
        self.index2word = index2word
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.SOS_token = 1
        self.EOS_token = 2
        self.use_cuda = torch.cuda.is_available()

    def train(self, input_variables, target_variables, criterion, lm_optimizer):
        ''' Pad all tensors in this batch to same length. '''
        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)

        batch_size = input_variables.size()[1]
        target_length = target_variables.size()[0]

        lm_optimizer.zero_grad()
        loss = 0

        lm_inputs = input_variables[0, :].view(1, -1)
        lm_hidden = self.lm.init_hidden(batch_size)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                lm_outputs, lm_hidden = self.lm(lm_inputs, lm_hidden)
                loss += criterion(lm_outputs, target_variables[di, :])
                lm_inputs = target_variables[di, :].view(1, -1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                lm_outputs, lm_hidden = self.lm(lm_inputs, lm_hidden)
                topv, topi = lm_outputs.data.topk(1)
                lm_inputs = topi.permute(1, 0)
                loss += criterion(lm_outputs, target_variables[di])

        loss.backward()
        lm_optimizer.step()

        return loss.item() / target_length

    def evaluate(self, input_variables, seed_length):
        with torch.no_grad():
            ''' Pad all tensors in this batch to same length. '''
            input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)

            if self.use_cuda: input_variables = input_variables.cuda()

            target_length = input_variables.size()[0]
            batch_size = input_variables.size()[1]
            lm_inputs = input_variables[0, :].view(1, -1)
            lm_hidden = self.lm.init_hidden(batch_size)

            output_words = [[] for i in range(batch_size)]

            for di in range(target_length):
                lm_outputs, lm_hidden = self.lm(lm_inputs, lm_hidden)

                topv, topi = lm_outputs.data.topk(1)
                for i, ind in enumerate(topi[0]):
                    output_words[i].append(self.index2word[ind])

                if di+1 < seed_length:
                    lm_inputs = input_variables[di+1, :].view(1, -1)
                else:
                    lm_inputs = topi.permute(1, 0)
                    if self.use_cuda: lm_inputs = lm_inputs.cuda()

        return output_words
