import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Dense, Dropout, Reshape, Flatten
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.optimizers import Adadelta, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec

class CNN:
    def __init__(self, x, y, data_features, data_sequence, num_labels):
        self.x = x
        self.y = y
        self.num_labels = num_labels
        self.data_features = data_features
        self.data_sequence = data_sequence
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=((self.data_sequence, self.data_features))))

    def graph(self, type='shallow', filter_size=3, filter_num=100, num_filter_block=None, dropout_rate=0):
        if type == 'shallow':
            self.model.add(Conv1D(kernel_size=filter_size, strides=1, filters=filter_num, padding='valid', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=self.data_sequence - filter_size, strides=1, padding='valid'))
            self.model.add(Flatten())
            self.model.add(Dropout(rate=dropout_rate))
            self.model.add(Dense(self.num_labels, activation='softmax'))
        elif type == 'deep':
            self.model.add(Conv1D(kernel_size=kernel_size, filters=filter_num, padding='valid', activation='relu'))

            cur_filter_num = filter_num
            for _, num_blocks in enumerate(num_filter_block):
                for i in range(num_blocks):
                    ConvBlock(model=self.model, kernel_size=filter_size, filters=cur_filter_num)
                self.model.add(MaxPooling1D(pool_size=3, strides=2))
                cur_filter_num *= 2
            self.model.add(Dense(2048, activation='relu'))
            self.model.add(Dense(2048, activation='relu'))
            self.model.add(Dense(self.num_labels, activation='softmax'))

        def ConvBlock(model, filters, kernel_size=filter_size):
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))

    def train(self, optimizer, modelpath, num_epochs=1, mini_batch=1, val_split=0.2):
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit(self.x, self.y, validation_split=val_split, epochs=num_epochs, 
                       batch_size=mini_batch, callbacks=callbacks_list, verbose=0)

def load_data(filepath, embeddings_path):
    yelp_data = pd.read_csv(filepath, header=None)
    yelp_x = yelp_data.iloc[:,1:].values
    y = yelp_data.iloc[:,0].values-1 # fix since output labels are bizarly {1,2}

    model = Word2Vec.load(embeddings_path)
    x = []
    data_sequence = 0
    for i, s in enumerate(yelp_x):
        sample = []
        for w in re.sub('[^a-zA-Z0-9\s]', '', s[0]).split():
            try:
                sample.append(list(model[re.sub('\W', '', w[0].lower())]))
                x.append(sample)
                if len(sample) > data_sequence:
                    data_sequence = len(sample)
            except KeyError:
                pass
        if i % 1000 == 0:
            print('Loading: %6d/%d' % (i, len(yelp_x)), end='\r', flush=True)
    print('Loading: %6d/%d' % (i, len(yelp_x)))

    zeros = [0.0 for i in range(300)]
    for i,v in enumerate(x):
        for _ in range(data_sequence - len(v)):
            x[i].append(zeros)

    print('Dataset shape: ', x.shape)
    return np.array(x), y, data_sequence

def main():
    # CONSTANTS
    data_features = 300
    data_num_classes = 2
    modelpath = 'model/yelp/scnn'

    # HYPERPARAMETERS
    filter_size = 3
    filter_num = 100
    dropout_rate = 0.5
    mini_batch = 50
    num_epochs = 1

    print('Loading Dataset')
    x, y, data_sequence = load_data('data/csv/yelp_dataset/train.csv', 'data/word2vec/yelp_combined_word2vec')
    print('Training Model')
    shallow= CNN(x, y, data_features=data_features, data_sequence=data_sequence, num_labels=data_num_classes)
    shallow.arch(dropout_rate=dropout_rate)
    shallow.train(optimizer=Adadelta(), modelpath=modelpath, num_epochs=num_epochs, mini_batch=mini_batch)

if __name__ == '__main__':
    main()

