import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, Flatten,  Dense
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.python.keras.optimizers import Adadelta

'''
CONSTANTS
Input is a sequence of word2vec vectors representing a single sentence/training instance.
Input is padded with zero vectors if sentence length < max(len(sentence)).
Input is expected flattened.
'''
data_vector_size = 300
data_vector_num = 60
data_shape = (data_vector_size, data_vector_num)
data_size_flat = np.prod(data_shape)
data_num_classes = 2
path_to_model = './model'

'''
HYPERPARAMETERS
'''
filter_size = 3
filter_num = 100
dropout_rate = 0.5
mini_batch = 50
num_epoch = 1

'''
MODEL
'''
model = Sequential()
model.add(InputLayer(input_shape=(data_size_flat,)))
model.add(Reshape(data_shape))
model.add(Conv1D(kernel_size=3, strides=1, filters=100, padding='valid', activation='relu'))
model.add(MaxPooling1D(pool_size=data_vector_num - filter_size, strides=1, padding='valid'))
model.add(Flatten())
model.add(Dropout(rate=dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adadelta()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=mini_batch)
result = model.evaluate(x=x_valid, y=y_valid)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
model.save(path_to_model)
