import pandas as pd
import numpy as np
import re
import os
import pickle
from gensim.models import word2vec
from sklearn.cluster import KMeans
import random
import logging

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec, doc2vec
from gensim.models import word2vec

# Mandarin(Chinese) article preprocess & jieba words cut
# Saved article with word cutted as pickle file, for easy asscess as list format later on
# discussion website article dataset columns = id, title, content, date, count like, count dislike 
def article_preprocess(raw_article):
    article = pd.read_csv(raw_article)
    
    # preprocess rules
    article['content'] = article['content'].str.replace('https?:\/\/\S*', '')
    article['content'] = article['content'].replace('', np.nan)
    # remove nan
    article = article.dropna()
    article = article.reset_index(drop=True)
    article['idx'] = article.index
    preprocess_article = article
    
    return preprocess_article


# word cut with jieba dictionary
def jieba_word_cut(jieba_dic, stop_words, article, saved_path):
    
    jieba.set_dictionary(jieba_dic)
    stop_words = open(stop_words).read().splitlines()
    data = article['content'].tolist()
    
    sentences = []
    for i, text in enumerate(data):
        line = []
        for w in jieba.cut(text, cut_all=False):
            ## set rules, ex.remove stopwords and digits
            if w not in stop_words and not bool(re.match('[0-9]+', w)):
                line.append(w)
        sentences.append(line)
        
    # saved article_cutted as pickle file, for easy asscess as list format        
    with open(saved_path, "wb") as file:
        pickle.dump(sentences, file)
    
    article_cutted = sentences
    return article_cutted


# Separate like & dislike article from article_preprocess by threshold
def threshold(df, sentences, diff_threshold):
    df = df[abs(df['like']-df['dislike']) > diff_threshold].copy()
    df['type'] = np.clip(df['like']-df['dislike'], 0, 1)
    df = df.reset_index(drop=True)
    print(df['type'].value_counts())
    return df


# main()
raw_article = 'raw_article_file_path'
jieba_dic = 'jieba_dic_file_path'
stop_words = 'stop_words_list_file_path'
saved_path = 'saved_path_file_path'
diff_threshold = 20

article = article_preprocess(raw_article)
article_cutted = jieba_word_cut(jieba_dic, stop_words, article, saved_path)
liked_article = threshold(article, article_cutted, diff_threshold)


# bag_of_words & tf_idf

sentences = cut_sentences  

def bag_of_words(sentences):
    
    with open("article_cutted", "rb") as file:
        sentences = pickle.load(file)
    # define transformer
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform([' '.join(x) for x in sentences])
    # saved as pickle file
    with open("article_count", "wb") as file:
        pickle.dump([vectorizer, count], file)

    # select top 10 frequency of words
    # create a dictionary: id as key ; word as values
    id2word = {v:k for k, v in vectorizer.vocabulary_.items()}    
    # columnwise sum: words frequency
    sum_ = np.array(count.sum(axis=0))[0]      
    # top 10 frequency's wordID
    most_sum_id = sum_.argsort()[::-1][:10].tolist()   
    # print top 10 frequency's words
    features = [id2word[i] for i in most_sum_id]
    data = pd.DataFrame(count[df.idx.as_matrix(),:][:,most_sum_id].toarray(), columns=features)

    # compute correlation
    data = pd.concat([df.type, data], axis=1)
    return data.corr()
    
       
def tf_idf(sentences):
    
    with open("article_cutted", "rb") as file:
        sentences = pickle.load(file)
    # define transformer 
    vectorizer = TfidfVectorizer(norm=None) # do not do normalize
    tfidf = vectorizer.fit_transform([' '.join(x) for x in sentences])
    # saved data as pickle format
    with open("article_tfidf", "wb") as file:
        pickle.dump([vectorizer, tfidf], file)
        
    # create a dictionary: id as key ; word as values
    id2word = {v:k for k, v in vectorizer.vocabulary_.items()}
    # columnwise average: words tf-idf
    avg = tfidf.sum(axis=0) / (tfidf!=0).sum(axis=0)

    # set df < 20 as 0
    avg[(tfidf!=0).sum(axis=0)<20] = 0
    avg = np.array(avg)[0]
    # top 10 tfidf's wordID
    most_avg_id = avg.argsort()[::-1][:10].tolist()
    # print top 10 tf-idf's words
    features = [id2word[i] for i in most_avg_id]

    data = pd.DataFrame(tfidf[df.idx.as_matrix(),:][:,most_avg_id].toarray(), columns=features)

    # compute correlation
    data = pd.concat([df.type, data], axis=1)
return data.corr()

#word2vec.main()
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

    
# main()
sg = 0
vec_size = 256
min_count_of_each_word = 5
window_size = 5
n_epoch = 5
word2vec(sg, vec_size, min_count_of_each_word, window_size, n_epoch)


###############  Application of word2vec  ###############

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
        
# clustering

# create a dictionary: words as key ; count as values
words = {word: vocab.count for word, vocab in model.wv.vocab.items()}
# sort and select the top 10000 count of words
words = sorted(words.items(), key=lambda x: x[1], reverse=True)
words = words[:10000]
words = np.array(words)[:, 0]

# extract the word vectors 
vecs = model.wv[words]
# run clustering algorithm
kmeans = KMeans(n_clusters=50)
cluster = kmeans.fit_predict(vecs)

# print the result
df = pd.DataFrame([words.tolist(), cluster.tolist()], index=['words', 'no. cluster']).T
df.head(n=5)

# print every cluster of words
data = pd.concat([d['words'].reset_index(drop=True).rename(columns={0: k}) for k, d in df.groupby('no. cluster')], axis=1)
data

# end of word to vector


# doc_to_vec.main()
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

# end of document to vector