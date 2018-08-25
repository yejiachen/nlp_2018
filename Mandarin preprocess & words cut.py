# Mandarin(Chinese) article preprocess & jieba words cut
# Saved article with word cutted as pickle file, for easy asscess as list format later on

import pandas as pd
import numpy as np
import re
import os
import pickle
import jieba
import jieba.posseg as pseg

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
    
    return article


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

        
# main()
raw_article = 'raw_article_file_path'
jieba_dic = 'jieba_dic_file_path'
stop_words = 'stop_words_list_file_path'
saved_path = 'saved_path_file_path'

article = article_preprocess(raw_article)
jieba_word_cut(jieba_dic, stop_words, article, saved_path)
