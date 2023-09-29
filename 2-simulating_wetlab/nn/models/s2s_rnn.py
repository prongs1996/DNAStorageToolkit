'''
Implementation of seq2seq RNN models and its variants.
Author: Longshen Ou

Implementation of AttentionalRnnDecoder is adapted from SpeechBrain Toolkit: 
https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.RNN.html#speechbrain.nnet.RNN.AttentionalRNNDecoder
'''

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _main():
    check_s2s()
    # check_sb_decoer()


def check_encoder():
    vocab_size = 8
    d_model = 64
    n_layer = 1

    enc_dim = d_model
    enc_states = torch.rand([4, 100, enc_dim])  # [B, L, d]
    inp_tensor = torch.rand([4, 10, vocab_size])  # [B, L, D]

    enc = EncoderRNN(
        input_size=vocab_size,
        hidden_size=d_model,
    )
    out = enc(inp_tensor)
    print(out.shape)


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
        bi_enc=True,
    )
    out, attn = model(inp_tensor, dec_inp)
    print(out.shape, attn.shape)


def check_torch_s2s():
    hidden_size = 10
    vocab_size = 6
    encoder = EncoderRNN(vocab_size, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, vocab_size)
    # input_length = input_tensor.size(0)

    max_length = 15
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    encoder_hidden = torch.zeros(1, 1, hidden_size, device=device)

    input_tensor = torch.tensor([[1, 3, 5], [0, 2, 4]])
    # for i in range(input_length): # input_tensor[i]
    encoder_output, encoder_hidden = encoder(input_tensor)
    encoder_outputs = encoder_output

    SOS_token = 2
    decoder_input = torch.tensor([[SOS_token], [SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    target_length = 7
    target_tensor = torch.tensor([3, 2, 1, 0])

    # teacher forcing
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        # loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing


def check_sb_decoer():
    d_model = 64
    n_layer = 1

    enc_dim = d_model
    enc_states = torch.rand([4, 100, enc_dim])  # [B, L, d]
    inp_tensor = torch.rand([4, 5, 64])  # [B, L, D]

    dec = AttentionalRNNDecoder(
        rnn_type='gru',
        attn_type='content',
        hidden_size=d_model,  # output dim
        attn_dim=d_model,  # attention dim
        num_layers=n_layer,
        enc_dim=enc_dim,  # encoder output dim
        input_size=d_model,  # input dim
    )
    out, attn = dec(inp_tensor, enc_states)
    print(out.shape)  # [4, 6, 64]


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    EOS_token = '</s>'
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_lang = 'en'
    output_lang = 'ch'
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def _procedures():
    pass


class Seq2seqRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(
            vocab_size), int(hidden_size), int(enc_rnn_layer), int(dec_rnn_layer)
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = EncoderRNN(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)
        self.decoder = AttentionalRNNDecoder(
            rnn_type='gru',
            attn_type='content',
            hidden_size=hidden_size,  # output dim
            attn_dim=hidden_size,  # attention dim
            num_layers=dec_rnn_layer,
            enc_dim=hidden_size if bi_enc == False else hidden_size*2,  # encoder output dim
            input_size=hidden_size,  # input dim
            emb_layer=self.embed,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, enc_inp, dec_inp):
        enc_out, _ = self.encoder(enc_inp)
        out, attn = self.decoder(dec_inp, enc_out)
        out = self.lm_head(out)
        return out, attn


class S2sCRNN(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(
            vocab_size), int(hidden_size), int(enc_rnn_layer), int(dec_rnn_layer)
        self.encoder = EncoderCRNN(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)


class S2sResCRNN(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(
            vocab_size), int(hidden_size), int(enc_rnn_layer), int(dec_rnn_layer)
        self.encoder = EncoderResCRNN(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)


class S2sSARNN(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(vocab_size), int(hidden_size), int(
            enc_rnn_layer), int(dec_rnn_layer)
        self.encoder = EncoderSARNN(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)


class S2sSARNN2(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(vocab_size), int(hidden_size), int(
            enc_rnn_layer), int(dec_rnn_layer)
        self.encoder = EncoderSARNN2(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)


class S2sResSARNN(Seq2seqRNN):
    def __init__(self, vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc):
        super().__init__(vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer, bi_enc)
        vocab_size, hidden_size, enc_rnn_layer, dec_rnn_layer = int(vocab_size), int(hidden_size), int(
            enc_rnn_layer), int(dec_rnn_layer)
        self.encoder = EncoderResSARNN(
            self.embed, hidden_size, num_rnn_layer=enc_rnn_layer, bi=bi_enc)


class EncoderRNN_old(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        '''
        input: [B, L]
        '''
        embedded = self.embedding(input)  # [B, L, D]
        # gru input should be [B, L, D_in], hidden: [#layer, B, D_h]
        output = embedded
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        output, hidden = self.gru(output)
        return output, hidden


class EncoderRNN(nn.Module):
    def __init__(self, emb_layer, hidden_size, num_rnn_layer, bi):
        super().__init__()
        self.emb = emb_layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_rnn_layer,
            bidirectional=bi
        )

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        # output = embedded # gru input should be [B, L, D_in], hidden: [#layer, B, D_h]
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


class EncoderCRNN(nn.Module):
    def __init__(self, emb_layer, hidden_size, num_rnn_layer, bi):
        super().__init__()
        self.emb = emb_layer
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_rnn_layer,
            bidirectional=bi
        )

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


class EncoderResCRNN(EncoderCRNN):

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        x = x.permute(0, 2, 1)

        y = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x) + y

        y = x.clone()
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x) + y

        x = x.permute(0, 2, 1)
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


class EncoderSARNN2(nn.Module):
    def __init__(self, emb_layer, hidden_size, num_rnn_layer, bi):
        super().__init__()
        self.emb = emb_layer
        self.sa1 = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=2, dim_feedforward=1024)
        # nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.sa2 = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=2, dim_feedforward=1024)
        # self.norm1 = nn.LayerNorm(hidden_size)
        # self.norm2 = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_rnn_layer,
            bidirectional=bi
        )

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        x = self.sa1(x)
        x = self.sa2(x)
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


class EncoderSARNN(nn.Module):
    def __init__(self, emb_layer, hidden_size, num_rnn_layer, bi):
        super().__init__()
        self.emb = emb_layer
        self.sa1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.sa2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_rnn_layer,
            bidirectional=bi
        )

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        x, _ = self.sa1(x, x, x)
        x = self.norm1(x)
        x, _ = self.sa2(x, x, x)
        x = self.norm2(x)
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


class EncoderResSARNN(nn.Module):
    def __init__(self, emb_layer, hidden_size, num_rnn_layer, bi):
        super().__init__()
        self.emb = emb_layer
        self.sa1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.sa2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_rnn_layer,
            bidirectional=bi
        )

    def forward(self, x):
        '''
        input: [B, L]
        '''
        x = self.emb(x)  # [B, L, D]
        y = x.clone()
        x, _ = self.sa1(x, x, x)
        x = self.norm1(x + y)
        y = x.clone()
        x, _ = self.sa2(x, x, x)
        x = self.norm2(x + y)
        # hidden: [layer=1, B=2, D_h=10]  out:[B=2, L=3, D_h(x2)=10]
        x, hidden = self.gru(x)
        return x, hidden


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class ContentBasedAttention(nn.Module):
    """ This class implements content-based attention module for seq2seq
    learning.

    Reference: NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN
    AND TRANSLATE, Bahdanau et.al. https://arxiv.org/pdf/1409.0473.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = ContentBasedAttention(enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim, scaling=1.0):
        super(ContentBasedAttention, self).__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.

        """

        if self.precomputed_enc_h is None:
            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h)
        ).squeeze(-1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # compute context vectors
        # [B, 1, L] X [B, L, F]
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        context = self.mlp_out(context)

        return context, attn


class AttentionalRNNDecoder(nn.Module):
    """This function implements RNN decoder model with attention.

    This function implements different RNN models. It accepts in enc_states
    tensors formatted as (batch, time, fea). In the case of 4d inputs
    like (batch, time, fea, channel) the tensor is flattened in this way:
    (batch, time, fea*channel).

    Arguments
    ---------
    rnn_type : str
        Type of recurrent neural network to use (rnn, lstm, gru).
    attn_type : str
        type of attention to use (location, content).
    hidden_size : int
        Number of the neurons.
    attn_dim : int
        Number of attention module internal and output neurons.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    input_shape : tuple
        Expected shape of an input.
    input_size : int
        Expected size of the relevant input dimension.
    nonlinearity : str
        Type of nonlinearity (tanh, relu). This option is active for
        rnn and ligru models only. For lstm and gru tanh is used.
    re_init : bool
        It True, orthogonal init is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    normalization : str
        Type of normalization for the ligru model (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    scaling : float
        A scaling factor to sharpen or smoothen the attention distribution.
    channels : int
        Number of channels for location-aware attention.
    kernel_size : int
        Size of the kernel for location-aware attention.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).

    Example
    -------
    >>> enc_states = torch.rand([4, 10, 20])
    >>> wav_len = torch.rand([4])
    >>> inp_tensor = torch.rand([4, 5, 6])
    >>> net = AttentionalRNNDecoder(
    ...     rnn_type="lstm",
    ...     attn_type="content",
    ...     hidden_size=7,
    ...     attn_dim=5,
    ...     num_layers=1,
    ...     enc_dim=20,
    ...     input_size=6,
    ... )
    >>> out_tensor, attn = net(inp_tensor, enc_states, wav_len)
    >>> out_tensor.shape
    torch.Size([4, 5, 7])
    """

    def __init__(
            self,
            rnn_type,
            attn_type,
            hidden_size,
            attn_dim,
            num_layers,
            enc_dim,
            input_size,  # decoder input dimention after embedding
            emb_layer,
            nonlinearity="relu",
            re_init=True,
            normalization="batchnorm",
            scaling=1.0,
            channels=None,
            kernel_size=None,
            bias=True,
            dropout=0.0,
    ):
        super(AttentionalRNNDecoder, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.attn_type = attn_type.lower()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.scaling = scaling
        self.bias = bias
        self.dropout = dropout
        self.normalization = normalization
        self.re_init = re_init
        self.nonlinearity = nonlinearity

        # only for location-aware attention
        self.channels = channels
        self.kernel_size = kernel_size

        # Combining the context vector and output of rnn
        self.proj = nn.Linear(
            self.hidden_size + self.attn_dim, self.hidden_size
        )

        assert self.attn_type == 'content'
        # if self.attn_type == "content":
        self.attn = ContentBasedAttention(
            enc_dim=enc_dim,
            dec_dim=self.hidden_size,
            attn_dim=self.attn_dim,
            output_dim=self.attn_dim,
            scaling=self.scaling,
        )
        # elif self.attn_type == "location":
        #     self.attn = LocationAwareAttention(
        #         enc_dim=enc_dim,
        #         dec_dim=self.hidden_size,
        #         attn_dim=self.attn_dim,
        #         output_dim=self.attn_dim,
        #         conv_channels=self.channels,
        #         kernel_size=self.kernel_size,
        #         scaling=self.scaling,
        #     )
        #
        # elif self.attn_type == "keyvalue":
        #     self.attn = KeyValueAttention(
        #         enc_dim=enc_dim,
        #         dec_dim=self.hidden_size,
        #         attn_dim=self.attn_dim,
        #         output_dim=self.attn_dim,
        #     )

        # else:
        #     raise ValueError(f"{self.attn_type} is not implemented.")

        self.drop = nn.Dropout(p=self.dropout)

        # set dropout to 0 when only one layer
        dropout = 0 if self.num_layers == 1 else self.dropout

        # using cell implementation to reduce the usage of memory
        assert self.rnn_type == 'gru'
        cell_class = GRUCell

        kwargs = {
            "input_size": input_size + self.attn_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bias": self.bias,
            "dropout": dropout,
            "re_init": self.re_init,
        }
        if self.rnn_type == "rnn":
            kwargs["nonlinearity"] = self.nonlinearity

        self.rnn = cell_class(**kwargs)
        self.emb = emb_layer

    def forward_step(self, inp, hs, c, enc_states, enc_len):
        """One step of forward pass process.

        Arguments
        ---------
        inp : torch.Tensor
            The input of current timestep.
        hs : torch.Tensor or tuple of torch.Tensor
            The cell state for RNN.
        c : torch.Tensor
            The context vector of previous timestep.
        enc_states : torch.Tensor
            The tensor generated by encoder, to be attended.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        dec_out : torch.Tensor
            The output tensor.
        hs : torch.Tensor or tuple of torch.Tensor
            The new cell state for RNN.
        c : torch.Tensor
            The context vector of the current timestep.
        w : torch.Tensor
            The weight of attention.
        """
        cell_inp = torch.cat([inp, c], dim=-1)
        cell_inp = self.drop(cell_inp)
        cell_out, hs = self.rnn(cell_inp, hs)

        c, w = self.attn(enc_states, enc_len, cell_out)
        dec_out = torch.cat([c, cell_out], dim=1)
        dec_out = self.proj(dec_out)

        return dec_out, hs, c, w

    def forward(self, inp_tensor, enc_states):
        """This method implements the forward pass of the attentional RNN decoder.

        Arguments
        ---------
        inp_tensor : torch.Tensor
            The input tensor for each timesteps of RNN decoder.
            Need to be embedded.
        enc_states : torch.Tensor
            The tensor to be attended by the decoder.
        wav_len : torch.Tensor
            This variable stores the relative length of wavform.

        Returns
        -------
        outputs : torch.Tensor
            The output of the RNN decoder.
        attn : torch.Tensor
            The attention weight of each timestep.
        """
        # calculating the actual length of enc_states
        # enc_len = torch.round(enc_states.shape[1] * wav_len).long()
        enc_len = torch.tensor([enc_states.shape[1]]).long()

        # embedding
        inp_tensor = self.emb(inp_tensor)

        # initialization
        self.attn.reset()
        c = torch.zeros(
            enc_states.shape[0], self.attn_dim, device=enc_states.device
        )
        hs = None

        # store predicted tokens
        outputs_lst, attn_lst = [], []
        for t in range(inp_tensor.shape[1]):
            outputs, hs, c, w = self.forward_step(
                inp_tensor[:, t], hs, c, enc_states, enc_len
            )
            outputs_lst.append(outputs)
            attn_lst.append(w)

        # [B, L_d, hidden_size]
        outputs = torch.stack(outputs_lst, dim=1)

        # [B, L_d, L_e]
        attn = torch.stack(attn_lst, dim=1)

        return outputs, attn


class GRUCell(nn.Module):
    """ This class implements a basic GRU Cell for a timestep of input,
    while GRU() takes the whole sequence as input.

    It is designed for an autoregressive decoder (ex. attentional decoder),
    which takes one input at a time.
    Using torch.nn.GRUCell() instead of torch.nn.GRU() to reduce VRAM
    consumption.
    It accepts in input tensors formatted as (batch, fea).

    Arguments
    ---------
    hidden_size: int
        Number of output neurons (i.e, the dimensionality of the output).
    input_shape : tuple
        The shape of an example input. Alternatively, use ``input_size``.
    input_size : int
        The size of the input. Alternatively, use ``input_shape``.
    num_layers : int
        Number of layers to employ in the GRU architecture.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    re_init : bool
        It True, orthogonal initialization is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 20])
    >>> net = GRUCell(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor, _ = net(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(
            self,
            hidden_size,
            input_shape=None,
            input_size=None,
            num_layers=1,
            bias=True,
            dropout=0.0,
            re_init=True,
    ):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size.")

        # Computing the feature dimensionality
        if input_size is None:
            if len(input_shape) > 3:
                self.reshape = True
            input_size = torch.prod(torch.tensor(input_shape[1:]))

        kwargs = {
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "bias": bias,
        }

        self.rnn_cells = nn.ModuleList([torch.nn.GRUCell(**kwargs)])
        kwargs["input_size"] = self.hidden_size

        for i in range(self.num_layers - 1):
            self.rnn_cells.append(torch.nn.GRUCell(**kwargs))

        self.dropout_layers = nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for _ in range(self.num_layers - 1)]
        )

        if re_init:
            rnn_init(self.rnn_cells)

    def forward(self, x, hx=None):
        """Returns the output of the GRUCell.

        Arguments
        ---------
        x : torch.Tensor
            The input of GRUCell.
        hx : torch.Tensor
            The hidden states of GRUCell.
        """

        # if not provided, initialized with zeros
        if hx is None:
            hx = x.new_zeros(self.num_layers, x.shape[0], self.hidden_size)

        h = self.rnn_cells[0](x, hx[0])
        hidden_lst = [h]
        for i in range(1, self.num_layers):
            drop_h = self.dropout_layers[i - 1](h)
            h = self.rnn_cells[i](drop_h, hx[i])
            hidden_lst.append(h)

        hidden = torch.stack(hidden_lst, dim=0)
        return h, hidden


def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.

    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)


if __name__ == '__main__':

    MAX_LENGTH = 3
    device = 'cpu'
    _main()
