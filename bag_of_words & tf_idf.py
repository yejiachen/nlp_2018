from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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