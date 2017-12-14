from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras import optimizers
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
import numpy as np
import random
import csv
import os
from os.path import dirname
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

random.seed(1882)

# =========================================
# LOADING DATA / DIFFERENT FOR EACH DATASET
# =========================================

def load_data_fakenews(filename):
    print("Loading Fake News data ... ")

    x = []
    y = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            if row[3] == 'FAKE':
                row[3] = 0
            if row[3] == 'REAL':
                row[3] = 1

            x.append(row[1] + row[2])
            y.append(row[3])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    return x_train, y_train, x_test, y_test

def load_data_yelp(filename, x, y):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x.append(row[1])
            if row[0] == '1':
                y.append(0)
            else:
                y.append(1)
    return x, y

def create_dictionaries(x, y):
    values = x
    keys = y
    data = dict(zip(keys, values))
    return data

def transform_data(model, x_train, y_train, x_test, y_test):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k+1 for k,v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}

    def parse_data(x,y):

        for key in range(len(y)):
            txt = x[key].lower().replace('\n', '').split()
            new_txt = []
            for word in txt:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            x[key] = new_txt
        return x,y

    x_train, y_train = parse_data(x_train, y_train)
    x_test, y_test = parse_data(x_test, y_test)

    return w2indx, w2vec, x_train, y_train, x_test, y_test

# ==============
# WORD2VEC MODEL
# ==============

def review_to_wordlist(x):
    words = x.lower().split()
    return words


def review_to_sentences(x, tokenz):
    print("Loading the word representation ...")
    raw_sentences = tokenz.tokenize(x.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences


def tokenizer(text):
    text = [document.lower().replace('\n', '').split() for document in text]
    return text

# ===========
# LSTM MODEL
# ===========


def lstm_model(X_train, y_train, X_test, y_test, vocab_dim, n_symbols, embedding_weights, input_length, output_name):

    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(units=512))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    print('Compiling the Model...')
    sgd = optimizers.rmsprop(lr=0.01, clipnorm=0.5)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    print("Train...")
    checkpoint = ModelCheckpoint(output_name, monitor='val_acc', mode='auto', save_best_only=True, verbose=1)
    callback_list = [checkpoint]
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test),
              shuffle=True, callbacks=callback_list)

    print("Evaluate...")
    score = model.evaluate(X_test, y_test, batch_size=32)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def main():

    # =======================
    # LOAD THE DATA FOR YELP
    # =======================

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_yelp = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/yelp_dataset/train.csv')
    test_yelp = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/yelp_dataset/test.csv')
   
    print("Loading Yelp data ... ")
    x_train, y_train = load_data_yelp(train_yelp, x_train, y_train)
    x_test, y_test = load_data_yelp(test_yelp, x_test, y_test)

    combined_x = x_train + x_test

    print("Tokenizing ...")
    combined_x = tokenizer(combined_x)

    # ==================================
    # CONVERT TO WORD2VEC REPRESENTATION
    # ==================================

    # Set parameters
    vocab_dim = 300
    n_exposures = 30
    window_size = 7
    maximum_string = max(combined_x, key=len)
    input_length = len(maximum_string)   # average length is 140, max is 1052

    print("Loading Yelp Word2Vec model ...")
    '''
    model = Word2Vec(size = vocab_dim, min_count = n_exposures, window = window_size)
    model.build_vocab(combined_x)
    model.train(combined_x, total_examples=model.corpus_count, epochs=model.iter)
    model.save("yelp_combined_word2vec")
    '''
    model = Word2Vec.load(os.path.join(dirname(dirname(os.getcwd())), 'data/word2vec/yelp_combined_word2vec'))

    # ===================
    # LSTM MODEL FOR YELP
    # ===================

    print("Transform the data ...")
    index_dict, word_vectors, x_train, y_train, x_test, y_test = transform_data(model, x_train, y_train,
                                                                                x_test, y_test)

    print("Setting up arrays for Neural Network Embedding Layer ... ")
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))

    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    print("Initializing Datasets ...")
    X_train = x_train
    y_train = y_train
    X_test = x_test
    y_test = y_test

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=input_length)
    X_test = sequence.pad_sequences(X_test, maxlen=input_length)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Convert labels to Numpy Sets...')
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Running the model ...")
    lstm_model(X_train, y_train, X_test, y_test, vocab_dim, n_symbols, embedding_weights, input_length, 'yelp_model.hdf5')


    # ==============================================================


    # ============================
    # LOAD THE DATA FOR FAKE NEWS
    # ============================

    news_data = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/fake_news/fake_news.csv')

    print("Loading Fake News data ... ")

    x_train_news, y_train_news, x_test_news, y_test_news = load_data_fakenews(news_data)

    combined_x_news = x_train_news + x_test_news

    print("Tokenizing Fake News data ...")

    combined_x_news = tokenizer(combined_x_news)


    print("Training a Fake News Word2Vec model ...")
    '''
    model = Word2Vec(size=vocab_dim, min_count=n_exposures, window=window_size)
    model.build_vocab(combined_x_news)
    model.train(combined_x_news, total_examples=model.corpus_count, epochs=model.iter)
    model.save("fakenews_combined_word2vec")
    '''
    model = Word2Vec.load(os.path.join(dirname(dirname(os.getcwd())), 'data/word2vec/fakenews_combined_word2vec'))
    
    # Set parameters
    vocab_dim = 300
    maximum_string = max(combined_x_news, key=len)
    input_length = len(maximum_string) # average length is 775

    # ========================
    # LSTM MODEL FOR FAKE NEWS
    # ========================

    print("Transform the data ...")
    index_dict, word_vectors, x_train, y_train, x_test, y_test = transform_data(model, x_train_news, y_train_news, x_test_news, y_test_news)

    print("Setting up arrays for Neural Network Embedding Layer ... ")
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    print("Initializing Datasets ...")
    X_train = x_train
    y_train = y_train
    X_test = x_test
    y_test = y_test

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen= input_length)
    X_test = sequence.pad_sequences(X_test, maxlen= input_length)

    print('Convert labels to Numpy Sets...')
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print("Running the model ...")

    lstm_model(X_train, y_train, X_test, y_test, vocab_dim, n_symbols, embedding_weights, input_length, 'fakenews_model.hdf5')


if __name__ == '__main__':
    main()

