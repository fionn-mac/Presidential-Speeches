import argparse

import torch

from data import Data
from embedding_google import Get_Embedding
from language_model import Language_Model
from train_network import Train_Network
from run_iterations import Run_Iterations

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations over the training set.", default=15)
    parser.add_argument("-nl", "--num_layers", type=int, help="Number of layers in Language Model.", default=2)
    parser.add_argument("-z", "--hidden_size", type=int, help="LSTM Hidden State Size", default=1024)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser.", default=0.001)

    parser.add_argument("-l", "--max_length", type=int, help="Maximum Sentence Length.", default=20)
    parser.add_argument("-tp", "--tracking_seed", type='str', help="Track change in outputs for a particular seed.", default='None')
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='./Datasets/Obama/')
    parser.add_argument("-w", "--weights_file", type=str, help="Filename in which model weights would be saved.", default='obama_lm.pt')
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing word embeddings.", default='../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')

    args = parser.parse_args()

    print('Model Parameters:')
    print('Hidden Size                      :', args.hidden_size)
    print('Batch Size                       :', args.batch_size)
    print('Number of Layers                 :', args.num_layers)
    print('Max. input length                :', args.max_length)
    print('Learning rate                    :', args.learning_rate)
    print('Number of epochs                 :', args.num_iters)
    print('------------------------------------------------\n')

    print('Reading input data.')
    data = Data(args.dataset, max_length=args.max_length)

    print("Number of training Samples       :", len(data.train_in_seq))
    print("Number of validation Samples     :", len(data.val_in_seq))

    print('Creating Word Embedding.')

    ''' Use pre-trained word embeddings '''
    embedding = Get_Embedding(data.word2index, data.word2count, args.embedding_file)

    language_model = Language_Model(args.hidden_size, data.vocab_size, embedding.embedding_matrix,
                                    num_layers=args.num_layers, use_embedding=True, train_embedding=False)

    if use_cuda: language_model = language_model.cuda()

    print("Training Network.")

    train_network = Train_Network(language_model, data.index2word, num_layers=args.num_layers)

    run_iterations = Run_Iterations(train_network, data.x_train, data.y_train, data.index2word,
                                    args.batch_size, args.num_iters, args.learning_rate,
                                    tracking_seed=args.tracking_seed, val_in_seq=data.x_val, val_out_seq=data.y_val)

    run_iterations.train_iters()
    run_iterations.evaluate_randomly()

    torch.save(language_model.state_dict(), args.weights_file)
