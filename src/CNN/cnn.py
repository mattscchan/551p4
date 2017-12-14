import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.optimizers import Adadelta, SGD
from keras.callbacks import ModelCheckpoint

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
    shallow= CNN(x, y, data_features, data_sequence, num_labels, filepath)
    shallow.arch(dropout_rate=dropout_rate)
    shallow.train(optimizer=Adadelta(), filepath=filepath, num_epochs=num_epochs, mini_batch=mini_batch)
