import pickle
from gensim.models import word2vec
from sklearn.cluster import KMeans
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
    model.save('word2vec_model/CBOW.wv.syn0.npy')

    
# Application of word2vec
# load word2vec model
model = word2vec.Word2Vec.load('/word2vec_model/CBOW.wv.syn0.npy')
# get most similarity with given words
model.wv.most_similar('nvidia')
        # Print >>> 
        #[('GPU', 0.5550138354301453),
        # ('TPU', 0.5424560308456421),
        # ('Pro', 0.5173478126525879),
        # ('intel', 0.5163905620574951),
        # ('NVIDIA', 0.5157663226127625),
        # ('Intel', 0.5154422521591187),
        # ('PSV', 0.4950483441352844),
        # ('Panasonic', 0.4948265850543976),
        # ('R5', 0.4917067289352417),
        # ('處理器', 0.4880176782608032)]

# get most similarity with given words's relationship
model.wv.most_similar(positive=['nvidia', 'GPU'], negative=['VC'])
        # Print >>>
        #[('TPU', 0.5958899259567261),
        # ('晶片組', 0.5429255962371826),
        # ('處理器', 0.5254557728767395),
        # ('第八代', 0.521049976348877),
        # ('LCD', 0.5136227011680603),
        # ('Intel', 0.5124219655990601),
        # ('intel', 0.5123609304428101),
        # ('GTX', 0.5064011812210083),
        # ('DDR4', 0.5059893131256104),
        # ('CPU', 0.5017791986465454)]

# get most similarity with given words's relationship
model.wv.most_similar(positive=['nvidia', 'GPU'], negative=['google'])
        # Print >>>
        #[('晶片組', 0.6002845764160156),
        # ('DDR4', 0.5600212812423706),
        # ('VGA', 0.5388824939727783),
        # ('處理器', 0.5230287313461304),
        # ('ADATA', 0.5223461389541626),
        # ('Kingston', 0.5212363004684448),
        # ('第八代', 0.5204957723617554),
        # ('Intel', 0.5192126035690308),
        # ('CPU', 0.5113897919654846),
        # ('intel', 0.5091017484664917)]
