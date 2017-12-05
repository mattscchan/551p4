import argparse
import tensorflow as tf 
import numpy as np

RAND_SEED = int(0xCAFEBABE)

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_file(email, label):
	return tf.train.Example(
			features=tf.train.Features(
				feature={
					'email': _bytes_feature(email),
					'label': _int64_feature(label)
		})).SerializeToString()

def _write_tfrecords(filename, examples_list):
	writer = tf.python_io.TFRecordWriter(filename)

	for example in examples_list:
		writer.write(example)
	writer.close()

def main(args):
	examples_list = []
	labels_list = []

	np.random.seed(RAND_SEED)

	with open('./trec07p/full/index', 'r') as file:
		for line in file:
			label = line.split(' ')[0]
			if label == 'ham':
				labels_list.append(1)
			if label == 'spam':
				labels_list.append(0)


	for index in range(1, 75420):
		with open(args.filename + '_' + str(index) + '.txt', 'r', encoding='ISO-8859-1') as f:
			email = ''
			for line in f:
				email += line
			examples_list.append(_convert_file(email, labels_list[index-1]))

	np.random.shuffle(examples_list)

	test_examples = examples_list[0:7542]
	validation_examples = examples_list[7542:15083]
	train_examples = examples_list[15083:]

	_write_tfrecords('test.tfrecords', test_examples)
	_write_tfrecords('validation.tfrecords', validation_examples)
	_write_tfrecords('train.tfrecords', train_examples)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help="Provide input file name.")
	args = parser.parse_args()
	main(args)