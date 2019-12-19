from collections import Counter

import _pickle
import gensim
import numpy as np
import os
import torch
from config import Config

config = Config()


def build_word2id(file):
    """
    build vocabulary：{word: id}
    :param file: "train_data.txt"
    :return: None
    """
    if os.path.exists(config.word2id_path):
        print("Vocabulary Exists")
        return
    else:
        word2id = {"_PAD_": 0, "UNK": 1}
        id2word = {0: "_PAD_", 1: "UNK"}
        path = config.word2id_path
        with open(file) as f:
            for line in f.readlines():
                # remove "\n" and space
                sp = line.strip().split()
                for word in sp[:-1]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
                        id2word[len(word2id)] = word
        with open(path, "w") as f:
            for w in id2word.keys():
                f.write(id2word[w] + "\t")
                f.write(str(w))
                f.write("\n")
        print("Successfully Generate Vocabulary")


def load_word2id():
    """
    :return: word_to_id {word: id}
    """
    word_to_id = {}
    with open(config.word2id_path) as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = sp[1]
            word_to_id[word] = idx
        return word_to_id


def load_corpus(data_path, max_sen_len):
    """
    :param data_path: "train_data.txt"/"test_data.txt"
    :param max_sen_len: a pre-defined max length of the sequences
    :return: contents, label
    """
    word2id = load_word2id()
    contents, label = [], []
    with open(data_path) as f:
        for line in f.readlines():
            sp = line.strip().split()
            content = [int(word2id.get(w, 1)) for w in sp[:-1]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [int(word2id['_PAD_'])] * (max_sen_len - len(content))
            contents.append(content)
            label.append(sp[-1])
    contents = torch.from_numpy(np.asarray(contents)).type(torch.LongTensor)
    label = torch.from_numpy(np.asarray([int(l) for l in label])).type(torch.LongTensor)
    return contents, label


def batch_iter(x, y, batch_size):
    """
    :param x: data
    :param y: label
    :param batch_size: how many samples in one single batch
    :return: a batch of data
    """
    data_len = len(x)
    num_batch = int(data_len / batch_size)
    indices = np.random.permutation(np.arange(data_len))
    x = x[indices]
    y = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        yield x[start_id:end_id], y[start_id:end_id]


def load_glove(word2id):
    """
    :param word2id: word_to_id {word: id}
    :return: pre-trained word embeddings weights
    """
    print("Loading Glove=============")
    OOV = []
    f = open(config.glove_model_100d, 'rb')
    model = _pickle.load(f)
    weight = np.array(np.random.uniform(-1., 1., [len(word2id)+1, model.vector_size]))
    for word in word2id.keys():
        try:
            weight[int(word2id[word])] = model[word]
        except KeyError:
            OOV.append(word)
    print("Out of the vocabulary：", len(OOV))
    weight = torch.FloatTensor(weight)
    return weight
