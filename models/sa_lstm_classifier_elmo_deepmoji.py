"""
Created by Chenyang Huang, 2018
"""

import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math
import os


class BertSelfAttention(nn.Module):
    """
    Extracted from
    """
    def __init__(self, hidden_size):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = 16
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class SelfAttentive(nn.Module):
    def __init__(self, hidden_size, att_hops=1, att_unit=100, dropout=0.2):
        super(SelfAttentive, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(hidden_size, att_unit, bias=False)
        self.ws2 = nn.Linear(att_unit, att_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        # self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = att_hops

    def forward(self, rnn_out, mask):
        outp = rnn_out
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.reshape(-1, size[2])  # [bsz*len, nhid*2]
        mask = mask.squeeze(2)
        concatenated_mask = [mask for i in range(self.attention_hops)]
        concatenated_mask = torch.cat(concatenated_mask, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + concatenated_mask
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


class SelfAttention(nn.Module):
    """
    This implementation might not be good
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                     requires_grad=True)

        nn.init.xavier_uniform_(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):

        batch_size, max_len = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).view(-1, 1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 label_size, batch_size, att_mode="None", soft_last=True,
                 use_glove=True, add_linear=True, max_pool=False):
        super(AttentionLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bidirectional = True
        self.soft_last = soft_last
        self.num_layers = 3
        self.elmo_dim = 1024
        self.use_max_poll = max_pool
        if max_pool:
            self.max_pool = nn.MaxPool1d(3, stride=2)
            self.deepmoji_dim = 1151    # floor( (2304  - (k-1) - 1 )/ 2 + 1)
        else:
            self.deepmoji_dim = 2304

        self.deepmoji_out = 300
        if add_linear:
            self.deepmoji2linear = nn.Linear(self.deepmoji_dim, self.deepmoji_out)
        self.use_glove = use_glove
        self.add_lienar = add_linear
        if use_glove:
            self.input_dim = embedding_dim + self.elmo_dim
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.input_dim = self.elmo_dim

        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=0.75)
        # self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.5)
        # Select attention model
        if att_mode == 'bert':
            self.att = BertSelfAttention
        elif att_mode == 'attentive':
            self.att = SelfAttentive
        elif att_mode == 'self':
            self.att = SelfAttention
        elif att_mode == 'None':
            self.att = None

        if self.bidirectional:
            if self.att is not None:
                self.attention_layer = self.att(hidden_dim*2)
            if add_linear:
                self.hidden2label = nn.Linear(hidden_dim*2 + self.deepmoji_out, label_size)
            else:
                self.hidden2label = nn.Linear(hidden_dim*2 + self.deepmoji_dim, label_size)
        else:
            if self.att is not None:
                self.attention_layer = self.att(hidden_dim)
            if add_linear:
                self.hidden2label = nn.Linear(hidden_dim + self.deepmoji_out, label_size)
            else:
                self.hidden2label = nn.Linear(hidden_dim + self.deepmoji_dim, label_size)

        # self.last_layer = nn.Linear(hidden_dim, label_size * 100)
        # loss
        # weight_mask = torch.ones(vocab_size).cuda()
        # weight_mask[word2id['<pad>']] = 0
        # self.loss_criterion = nn.BCELoss()

    @staticmethod
    def sort_batch(batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        rever_sort = np.zeros(len(seq_lengths))
        for i, l in enumerate(perm_idx):
            rever_sort[l] = i
        return seq_tensor, seq_lengths, rever_sort.astype(int)

    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, x, seq_len, elmo, emoji_encoding):
        if self.use_glove:
            x, seq_len_sorted, reverse_idx = self.sort_batch(x, seq_len.view(-1))
            elmo, seq_len_sorted, reverse_idx = self.sort_batch(elmo, seq_len.view(-1))

            max_len = int(seq_len_sorted[0])
            embedded = self.embeddings(x)
            embedded = self.dropout(embedded)
            embedded = embedded[:, :max_len, :]

            elmo = self.dropout(elmo)
            combined_embd = torch.cat((embedded, elmo), dim=2)
        else:
            elmo, seq_len_sorted, reverse_idx = self.sort_batch(elmo, seq_len.view(-1))
            combined_embd = self.dropout(elmo)
        # embedded = torch.nn.BatchNorm1d(embedded)  # batch normalization
        packed_input = nn.utils.rnn.pack_padded_sequence(combined_embd, seq_len_sorted.cpu().numpy(), batch_first=True)
        hidden = self.init_hidden(x)
        packed_output, hidden = self.lstm(packed_input, hidden)
        lstm_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # global attention
        if self.att is not None:
            if isinstance(self.att, SelfAttention):
                out, att = self.attention_layer(lstm_out, unpacked_len)
            else:
                unpacked_len = [int(x.data) for x in unpacked_len]
                # print(unpacked_len)
                max_len = max(unpacked_len)
                mask = [[1] * l + [0] * (max_len - l) for l in unpacked_len]
                mask = torch.LongTensor(np.asarray(mask)).cuda()
                attention_mask = torch.ones_like(mask)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                out, alpha = self.attention_layer(lstm_out, extended_attention_mask)
                # out, att = self.attention_layer(lstm_out[:, -1:].squeeze(1), lstm_out)
                out = out[:, -1, :]
        else:
            output = lstm_out
            seq_len_tmp = torch.LongTensor(unpacked_len).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            seq_len_tmp = Variable(seq_len_tmp - 1).cuda()
            out = torch.gather(output, 1, seq_len_tmp).squeeze(1)

        # combined with deepmoji
        emoji_encoding, _, _ = self.sort_batch(emoji_encoding, seq_len.view(-1))
        if self.use_max_poll:
            emoji_encoding = self.max_pool(emoji_encoding.unsqueeze(1)).squeeze()  # max pooling
        if self.add_lienar:
            emoji_encoding = self.deepmoji2linear(emoji_encoding)
            emoji_encoding = F.relu(emoji_encoding)
            emoji_encoding = self.dropout(emoji_encoding)

        combined_out = torch.cat((out, emoji_encoding), dim=1)
        # loss = self.loss_criterion(nn.Sigmoid()(y_pred), y)
        y_pred = self.hidden2label(combined_out)

        y_pred = y_pred[reverse_idx]
        return y_pred

    def load_embedding(self, vectors):
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(vectors))
