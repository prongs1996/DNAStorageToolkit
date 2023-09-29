'''
Script contains interfaces of all model classes
Author: Longshen Ou
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.s2s_rnn import Seq2seqRNN, S2sCRNN, S2sResCRNN, S2sSARNN, S2sResSARNN, S2sSARNN2


def main():
    check_s2s()


def check_s2s():
    vocab_size = 5
    d_model = 64
    n_layer = 1

    enc_dim = d_model
    enc_states = torch.rand([4, 100, enc_dim])  # [B, L, d]
    inp_tensor = torch.randint(low=0, high=5, size=[4, 10])  # [B, L, D]
    dec_inp = torch.tensor([
        [1, 3, 4, 2],
        [1, 4, 3, 2],
        [1, 3, 3, 2],
        [1, 4, 4, 2],
    ])

    model = Seq2seqRNN(
        vocab_size=vocab_size,
        hidden_size=d_model,
        enc_rnn_layer=1,
        dec_rnn_layer=1,
    )
    out, attn = model(inp_tensor, dec_inp)
    print(out.shape, attn.shape)


def get_model(model_name, hparam):
    M = eval(model_name)
    print(model_name)
    model = M(
        **hparam['model_args']
    )
    return model


def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))


class Linear(nn.Module):
    def __init__(self, args):
        self.embed = nn.Embedding(4, 1024, padding_idx=0)
        self.linear = nn.Linear(1024, 4)


class s2s_naive(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc = nn.GRU


class S2sCrnn(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)


class MLP(nn.Module):
    '''
    Input and output length: 58
    '''

    def __init__(self, seq_len, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        seq_len, embed_dim, vocab_size, hidden_dim = int(seq_len), int(embed_dim), int(vocab_size), int(hidden_dim)
        # print(args['seq_len'])
        # print(args.seq_len)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.linear1 = nn.Linear(self.seq_len * self.embed_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.seq_len * self.vocab_size)

    def forward(self, x):
        '''
        x: [bs, seq_len]
        '''
        bs, seq_len = x.shape
        x = self.embed(x)  # [bs, seq_len, emb_len]
        x = x.reshape(bs, x.shape[1] * x.shape[2])  # [bs, seq_len * emb_len]
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)  # [bs, seq_len * vocab_size]
        x = x.reshape(bs, self.seq_len, self.vocab_size)

        # y = torch.zeros(size=(bs, seq_len, 5))
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         y[i][j][x[i][j]] = 1
        # x = y

        return x


if __name__ == '__main__':
    main()
