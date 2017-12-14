import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.optimizers import Adadelta, SGD
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
import random
import csv

# CONSTANTS
data_features = 300
data_sequence = 60
data_num_classes = 2
path_to_model = './model'

# HYPERPARAMETERS
filter_size = 3
filter_num = 100
dropout_rate = 0.5
mini_batch = 50
num_epochs = 1

class CNN:
    def __init__(self, x, y, data_features, data_sequence, num_labels):
        self.x = x
        self.y = y
        self.data_features = data_features
        self.data_sequence = data.sequence
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=((self.data_features, self.data_sequence))))

    def arch(self, type='shallow', filter_size=3, filter_num=100, num_filter_block=None, dropout_rate=0):
        if type == 'shallow':
            self.model.add(Conv1D(kernel_size=filter_size, strides=1, filters=filter_num, padding='valid', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=self.data_sequence - filter_size, strides=1, padding='valid'))
            self.model.add(Flatten())
            self.model.add(Dropout(rate=dropout_rate))
            self.model.add(Dense(num_classes, activation='softmax'))
        elif type == 'deep':
            model.add(Conv1D(kernel_size=kernel_size, filters=64, padding='valid', activation='relu'))

            cur_filter_num = filter_num
            for _, num_blocks in enumerate(num_filter_block):
                for i in range(num_blocks):
                    ConvBlock(model=self.model, kernel_size=filter_size, filters=cur_filter_num)
                model.add(MaxPooling1D(pool_size=3, strides=2))
                cur_filter *= 2
            model.add(Dense(2048, activation='relu'))
            model.add(Dense(2048, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

        def ConvBlock(model, kernel_size=filter_size, filters):
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))

    def train(self, optimizer, filepath, num_epochs=1, mini_batch=1, val_split=0.2):
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(self.x, self.y, validation_split=val_split, epochs=num_epochs, batch_size=mini_batch, callbacks=callbacks_list, verbose=0)

def main():
    # Load data
    yelp_path = 'data/yelp_dataset'
    fn_path = 'data/fakenews_dataset'
    trec_path = 'data/trec07p'
    
    yelp_df = ((pd.read_csv(yelp_path+'test.csv', header=None), pd.read_csv(yelp_path+'train.csv', header=None)), axis=0)
    yelp_x = yelp_df.ix[:,1:].values
    yelp_y = yelp_df.ix[:,0].values
    # Model

random.seed(1882)

# Data loading by Jean-Yves

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
            y.append(row[0])
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

def main():

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_yelp = "../csv/yelp_dataset/train.csv"
    test_yelp = "../csv/yelp_dataset/test.csv"

    print("Loading Yelp data ... ")
    x_train, y_train = load_data_yelp(train_yelp, x_train, y_train)
    x_test, y_test = load_data_yelp(test_yelp, x_test, y_test)

    combined_x = x_train + x_test

    print("Tokenizing ...")
    combined_x = tokenizer(combined_x)

    # Set parameters
    vocab_dim = 300
    n_exposures = 30
    maximum_string = max(combined_x, key=len)
    input_length = len(maximum_string)   # average length is 140, max is 1052

    print("Loading Yelp Word2Vec model ...")
    model = Word2Vec.load("yelp_combined_word2vec")

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
    lstm_model(X_train, y_train, X_test, y_test, vocab_dim, n_symbols, embedding_weights, input_length)

    # ==============================================================

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
    model = Word2Vec.load("fakenews_combined_word2vec")

    # Set parameters
    vocab_dim = 300
    maximum_string = max(combined_x_news, key=len)
    input_length = len(maximum_string) # average length is 775

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

    lstm_model(X_train, y_train, X_test, y_test, vocab_dim, n_symbols, embedding_weights, input_length)

    shallow= CNN(x, y, data_features, data_sequence, num_labels, filepath)
    shallow.arch(dropout_rate=dropout_rate)
    shallow.train(optimizer=Adadelta(), filepath=filepath, num_epochs=num_epochs, mini_batch=mini_batch)

if __name__ == '__main__':
    main()

