import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F

class Language_Model(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, num_layers=1, dropout_p=0.1,
                 use_embedding=False, train_embedding=True):
        super(Language_Model, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]

        self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        '''
        input           -> (1 x Batch Size)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        '''
        batch_size = input.size()[1]
        features = self.embedding(input) # (1, B, V)

        output, hidden = self.lstm(features, hidden)
        output = output.squeeze(0) # (1, B, V) -> (B, V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden

    def init_weights(self):
        ''' Initialize weights of lstm '''
        for name, param in self.lstm.named_parameters():
            if 'bias' in name_1:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_hidden(self, batch_size):
        # Hidden dimensionality : 2 (h_0, c_0) x Num. Layers * Num. Directions x Batch Size x Hidden Size
        result = torch.zeros(2, self.num_layers, batch_size, self.hidden_size)

        if self.use_cuda: return result.cuda()
        else: return result
