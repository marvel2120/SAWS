class Config:
    def __init__(self):
        super().__init__()
        self.word2id_path = "data/ghosh/word2id.txt"
        self.train_path = "data/ghosh/train.txt"
        self.test_path = "data/ghosh/test.txt"
        self.model_save_path = "model/ghosh/model.pt"
        self.train_loss_path = "loss_record/ghosh/train_loss_path.txt"
        self.test_loss_path = "loss_record//ghosh/test_loss_path.txt"
        self.max_sen_len = 40
        self.epoch = 200
        self.batch_size = 32
        self.embed_size = 100
        self.n_gram = 3
        self.num_filters = 100
        self.best_loss = float("inf")
        self.hit_patient = 0
        self.early_stop_patient = 20
        self.use_glove = True
        self.glove_path = "data/glove.6B.100d.txt"
        self.word2vec = "data/word2vec.txt"
        self.glove_model_100d = "data/pickled_model_100d"
