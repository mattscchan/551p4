import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import re
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Dense, Dropout, Reshape, Flatten, Embedding
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.optimizers import Adadelta, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary

class CNN:
    def __init__(self, x, y, data_features, data_sequence, num_labels, vocab_size, embeddings, savepath):
        self.x = x
        self.y = y
        self.num_labels = num_labels
        self.data_features = data_features
        self.data_sequence = data_sequence
        self.savepath = savepath
        if saved:
            print('Loading saved model')
            self.model = load_model(savepath)
        else:
            self.model = Sequential()
            #self.model.add(InputLayer(input_shape=((self.data_sequence, self.data_features))))
            self.model.add(Embedding(input_length=self.data_sequence,
                                     input_dim=vocab_size, 
                                     output_dim=self.data_features, 
                                     weights=[embeddings], 
                                     mask_zero=False,
                                     trainable=False))

    def graph(self, type='shallow', filter_size=3, filter_num=100, num_filter_block=None, dropout_rate=0):

        def ConvBlock(model, filters, kernel_size=filter_size):
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))

        if type == 'shallow' or type == 'scnn':
            self.model.add(Conv1D(kernel_size=filter_size, strides=1, filters=filter_num, padding='valid', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=self.data_sequence - filter_size, strides=1, padding='valid'))
            self.model.add(Flatten())
            self.model.add(Dropout(rate=dropout_rate))
            self.model.add(Dense(self.num_labels, activation='softmax'))

        elif type == 'deep' or type == 'dcnn':
            self.model.add(Conv1D(kernel_size=filter_size, filters=filter_num, padding='valid', activation='relu'))

            cur_filter_num = filter_num
            for _, num_blocks in enumerate(num_filter_block):
                for i in range(num_blocks):
                    ConvBlock(model=self.model, kernel_size=filter_size, filters=cur_filter_num)
                self.model.add(MaxPooling1D(pool_size=3, strides=2))
                cur_filter_num *= 2
            self.model.add(Dense(2048, activation='relu'))
            self.model.add(Dense(2048, activation='relu'))
            self.model.add(Dense(self.num_labels, activation='softmax'))

    def train(self, optimizer, num_epochs=1, mini_batch=1, val_split=0.2):
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit(self.x, self.y, validation_split=val_split, epochs=num_epochs, 
                       batch_size=mini_batch, callbacks=callbacks_list)

def load_data(filepath, embeddings_path, subset=None):
    yelp_data = pd.read_csv(filepath, header=None)
    yelp_x = yelp_data.iloc[:,1:].values
    yelp_y = yelp_data.iloc[:,0].values-1 # fix since output labels are bizarly {1,2}
    if not subset:
        subset = len(yelp_y)

    sequence_length = max([len(s.split(' ')) for s in yelp_x[:subset]])
    y = []
    for i in yelp_y[:subset]:
        if i == 0:
            y.append([1, 0])
        else:
            y.append([0, 1])

    model = Word2Vec.load(embeddings_path)
    vocabulary = Dictionary()
    vocabulary.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2idx = {v: k+1 for k,v in vocabulary.items()}
    w2vec = {word: model[word] for word in w2idx.keys()}

    x = []
    for i, s in enumerate(yelp_x[:subset]):
        sample = []
        for w in re.sub('[^a-zA-Z0-9\s]', '', s[0]).split():
            #try:
            sample.append(w2idx[re.sub('\W', '', w[0].lower())])
            #except KeyError:
            #    sample.append(0)
        # Padding
        if len(sample) > sequence_length:
            sample = sample[:sequence_length]
        else:
            for _ in range(sequence_length - len(sample)):
                sample.append(0)
        x.append(sample)
        if i % 1000 == 0:
            print('Loading: %6d/%d' % (i, subset), end='\r', flush=True)
    print('Loading: %6d/%d' % (i, subset))
    #print('X list ', len(x), len(x[0]))
    #print('Y list ', len(y), len(y[0]))

    x = np.array(x)
    y = np.array(y)
    print('Dataset shape: ', x.shape, y.shape)
    return x, y, w2idx, w2vec, sequence_length

def main(args):
    modelpath = 'model/' + args.dataset + '/'
    datapath = 'data/csv/' + args.dataset + '/train.csv'
    vecpath = 'data/word2vec/' + args.dataset + '_combined_word2vec'

    # CONSTANTS
    data_features = 300
    data_sequence = None
    data_num_classes = 2
    savepath = 'model/yelp/dcnn/{epoch:02d}-{val_acc:.2f}.hdf5'
    savepath = modelpath + args.model + '/best.hdf5'

    # HYPERPARAMETERS
    filter_size = 3
    if args.model == 'scnn':
        filter_num = 100
        dropout_rate = 0.5
        mini_batch = 50
        num_epochs = int(args.epochs)
        filter_blocks = None
        optimizer = Adadelta()
    else args.model == 'dcnn':
        filter_num = 64
        dropout_rate = 0
        mini_batch = 128
        num_epochs = int(args.epochs)
        filter_blocks = [10, 10, 4, 4]
        optimizer = SGD()

    print('Loading Dataset')
    x, y, idx, vec, data_sequence = load_data(datapath, vecpath, subset=subset)

    vocab_size = len(idx) + 1
    embeddings = np.zeros((vocab_size, data_features))
    for w, i in idx.items():
        embeddings[i, :] = vec[w]
    print('Embeddings shape: ', embeddings.shape)

    print('Training Model', args.model.upper() + '(' + args.dataset + ')')
    shallow= CNN(x=x, y=y, 
                 data_features=data_features, 
                 data_sequence=data_sequence, 
                 num_labels=data_num_classes, 
                 vocab_size=vocab_size, 
                 embeddings=embeddings,
                 savepath=savepath)
    if not args.saved:
        shallow.graph(type=args.model, 
                      num_filter_block=filter_blocks, 
                      dropout_rate=dropout_rate)
    shallow.train(optimizer=optimizer, 
                  modelpath=savepath, 
                  num_epochs=num_epochs, 
                  mini_batch=mini_batch)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('model')
    parser.add_argument('epochs', default=1)
    parser.add_argument('-s', '--saved', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

