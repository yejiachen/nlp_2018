# IMDB Reviews Sentiment Analysis, by using Bidirectional RNN

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Bidirectional


def preprocess(num_words, maxlen_of_article):
    # top (num_words) common words in English
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words)

    # set article lenth to maxlen for sequence limit
    x_train = sequence.pad_sequences(x_train, maxlen_of_article)
    x_test = sequence.pad_sequences(x_test, maxlen_of_article)
    
    return x_train, y_train, x_test, y_test

def main(num_words, dim, epochs, x_train, y_train, x_test, y_test):
    model = Sequential()
    # Set Embedding for one-hot encodding every words
    model.add(Embedding(num_words, dim))
    # Bidirectional rnn to do rnn both from front & end
    model.add(Bidirectional(GRU(units=dim, return_sequences=True)))
    model.add(Bidirectional(GRU(units=16)))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()

    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = 32, epochs = epochs)


    score = model.evaluate(x_test, y_test)
    print('validation loss:', score[0])
    print('validation accuracy:', score[1])
    
    
num_words = 10000
maxlen_of_article = 100
dim = 128
epochs = 5
x_train, y_train, x_test, y_test = preprocess(num_words, maxlen_of_article)
main(num_words, dim, epochs, x_train, y_train, x_test, y_test)

# end
# With 90% accuracy