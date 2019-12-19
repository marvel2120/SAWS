import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from config import Config

torch.manual_seed(123)
config = Config()


class Model(nn.Module):
    def __init__(self, word2id):
        super(Model, self).__init__()
        self.word2id = word2id
        self.vocab_size = len(self.word2id)
        self.embed_size = config.embed_size
        self.cnn = CNN_layer()
        self.phrase_attention = Phrase_attention()
        self.self_attention = Self_Attention()
        self.batch_size = config.batch_size
        self.embed_size = config.embed_size
        self.linear = nn.Linear(self.embed_size, 2)
        self.use_glove = config.use_glove
        self.uw = nn.Parameter(torch.FloatTensor(torch.randn(self.embed_size)))
        if self.use_glove:
            self.weight = utils.load_glove(self.word2id)
            self.embedding = nn.Embedding.from_pretrained(self.weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, x_batch):
        E = self.embedding(x_batch)
        U = self.cnn(E)
        a = self.phrase_attention(U).unsqueeze(2)
        f_a = self.self_attention(a * U)
        result = self.linear(f_a)
        return result


class CNN_layer(nn.Module):
    def __init__(self):
        super(CNN_layer, self).__init__()
        self.conv = nn.Conv2d(1, config.num_filters, (config.n_gram, config.embed_size))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, embedding):
        embedding = embedding.unsqueeze(1)
        embedding = self.conv(embedding)
        embedding = F.relu(embedding.squeeze(3))
        embedding = self.dropout(embedding)
        embedding = embedding.transpose(2, 1)
        return embedding


class Phrase_attention(nn.Module):
    def __init__(self):
        super(Phrase_attention, self).__init__()
        self.linear = nn.Linear(config.embed_size, config.max_sen_len - config.n_gram + 1)
        self.tanh = nn.Tanh()
        self.u_w = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(config.max_sen_len - config.n_gram + 1, 1)))

    def forward(self, embedding):
        u_t = self.tanh(self.linear(embedding))
        a = torch.matmul(u_t, self.u_w).squeeze(2)
        a = F.log_softmax(a, dim=1)
        return a


class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()
        self.w1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(config.embed_size, 1)))
        self.w2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(config.embed_size, 1)))
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(1)))

    def forward(self, embedding):
        f1 = torch.matmul(embedding, self.w1)
        f2 = torch.matmul(embedding, self.w2)
        f1 = f1.repeat(1, 1, embedding.size(1))
        f2 = f2.repeat(1, 1, embedding.size(1)).transpose(1, 2)
        S = f1 + f2 + self.b
        mask = torch.eye(embedding.size(1), embedding.size(1)).type(torch.ByteTensor)
        S = S.masked_fill(mask.bool().cuda(), -float('inf'))
        max_row = F.max_pool1d(S, kernel_size=embedding.size(1), stride=1)
        a = F.softmax(max_row, dim=1)
        v_a = torch.matmul(a.transpose(1, 2), embedding)
        return v_a.squeeze(1)