import pickle
from gensim.models import word2vec
import random
import logging
import os

# sg=0 CBOW ; sg=1 skip-gram

def word2vec(sg, vec_size, min_count_of_each_word, window_size, n_epoch):
    
    # Show training log & information
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    # load 'article_cutted'
    with open('article_cutted', 'rb') as file:
        data = pickle.load(file)
        
    # build word2vec
    # sg=0 CBOW ; sg=1 skip-gram
    model = word2vec.Word2Vec(size=vec_size, min_count=min_count_of_each_word, window=window_size, sg=sg)
    # build vocabulary
    model.build_vocab(data)
    # train word2vec model ; shuffle data every epoch
    for i in range(n_epoch):
        random.shuffle(data)
        model.train(data, total_examples=len(data), epochs=1)

    # save model
    model.save('word2vec_model/CBOW')