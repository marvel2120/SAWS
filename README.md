# SAWS
This repository is an implementation for paper "Modeling the Incongruity between Sentence Snippets for Sarcasm Detection".
To run the code, you just need to follow the below steps:
1. Due to the space limit, we didn't put the GloVe word embeddings here. You can download from  http://nlp.stanford.edu/projects/glove/
2. Put the file "glove.6B.100d.txt" under data directory.
3. Install all the packages which are required by the codes.
4. Run get_glove_model.py firstly to generate a word embedding vectors and save as pickle.
5. Edit the config.py file to change the hyper-parameters.
6. Type python main.py to train and test the model.
7. The model will be saved under model directory.
8. The training loss will be stored in loss_record folder.
