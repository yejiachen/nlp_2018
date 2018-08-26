import pickle
from gensim.models import Doc2Vec, doc2vec
from gensim.models import word2vec
import random
import numpy as np
import logging
import os

# sg=0 CBOW ; sg=1 skip-gram

def avg_word_vec(sg, vec_size, min_count_of_each_word, window_size, n_epoch):
    
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
    model.save('word2vec_model/CBOW.wv.syn0.npy')
    # load word2vec model
	model = word2vec.Word2Vec.load('word2vec_model/CBOW.wv.syn0.npy')

	# filter words that not in word2vec's vocab
	data_filtered = [[w for w in l if w in model.wv] for l in data]

	# compute average word vector
	avg_vector = []

	for l in data_filtered:
	    if len(l)==0:
	        avg_vector.append(np.array([0]*vec_size))
	    else:
	        avg_vector.append(np.mean([model.wv[w] for w in l], axis=0))

	# print result
	avg_vector[0]

	# save result
	with open('avg_article_vector', 'wb') as file:
	    pickle.dump(avg_vector, file)]


def doc2vec(vec_size, min_count_of_each_word, window_size, n_epoch):

	# load 'article_cutted'
	with open('article_cutted', 'rb') as file:
	    data = pickle.load(file)

	# create a document id map
	sentence_list = []
	for i, l in enumerate(data):
	    sentence_list.append(doc2vec.LabeledSentence(words=l, tags=[str(i)]))

	# define doc2vec model
	model = Doc2Vec(size=vec_size, min_count=min_count_of_each_word, window=window_size)
	# build vocabulary
	model.build_vocab(sentence_list)

	# train doc2vec model ; shuffle data every epoch
	for i in range(n_epoch):
	    random.shuffle(sentence_list)
	    model.train(sentence_list, total_examples=len(data), epochs=1)

	# print result
	model.docvecs['0']
	# save result
	model.save('word2vec_model/doc2vec.wv.syn0.npy')


# main()
vec_size = 256
min_count_of_each_word = 5
window_size = 5
n_epoch = 5
doc2vec(sg, vec_size, min_count_of_each_word, window_size, n_epoch)