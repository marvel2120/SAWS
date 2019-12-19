import _pickle
import gensim
import os
from config import Config
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

config = Config()
# read word embeddings and save as pickle
glove_file = datapath(os.getcwd() + "/" + config.glove_path)
tmp_file = get_tmpfile(os.getcwd() + "/" + config.word2vec)
glove2word2vec(glove_file, tmp_file)
model = gensim.models.KeyedVectors.load_word2vec_format(
    config.word2vec)
with open(config.glove_model_100d, "wb") as f:
    _pickle.dump(model, f)

# test
f = open(config.glove_model_100d, 'rb')
model = _pickle.load(f)
print(model.wv.similarity('have', 'has'))
print(model.most_similar("china"))
result = model.most_similar(positive=["women", "king"], negative=["man"])
print("{}:{:.4f}".format(*result[0]))
