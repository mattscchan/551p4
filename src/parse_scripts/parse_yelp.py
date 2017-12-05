import argparse
import csv
import tensorflow as tf 
from random import shuffle
import numpy as np

RAND_SEED = int(0xCAFEBABE)

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _create_example(review, label):
	return tf.train.Example(
			features=tf.train.Features(
				feature={
					'text': _bytes_feature(review),
					'label': _int64_feature(label)
		})).SerializeToString()

def _write_tfrecords(filename, examples_list):
	writer = tf.python_io.TFRecordWriter(filename)

	for example in examples_list:
		writer.write(example)
	writer.close()

def main(args):
	positive_examples = []
	negative_examples = []
	examples_all = []

	np.random.seed(RAND_SEED)

	with open(args.train, 'r') as f:
		csv_reader = csv.reader(f)
		count = 0
		for line in csv_reader:
			if line[0] == "1":
				negative_examples.append(_create_example(line[1], (int)(line[0])))
			if line[0] == "2":
				positive_examples.append(_create_example(line[1], (int)(line[0])))

			count += 1
			if count%10000 == 0:
				print(count)

	# Keep even class distribution for the Yelp Dataset train/validation
	np.random.shuffle(positive_examples)
	np.random.shuffle(negative_examples)

	validation_examples = positive_examples[0:28000] + negative_examples[0:28000]
	np.random.shuffle(validation_examples)
	_write_tfrecords('./data/validation.tfrecords', validation_examples)

	train_examples = positive_examples[28000:] + negative_examples[28000:]
	np.random.shuffle(train_examples)
	_write_tfrecords('./data/train.tfrecords', train_examples)

	with open(args.test, 'r') as f:
		count = 0
		csv_reader = csv.reader(f)
		for line in csv_reader:
			examples_all.append(_create_example(line[1], (int)(line[0])))

			count += 1
			if count%10000 == 0:
				print(count)

	np.random.shuffle(examples_all)
	_write_tfrecords('./data/test.tfrecords', examples_all)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train', help="Provide train file name.")
	parser.add_argument('test', help="Provide test file name.")
	args = parser.parse_args()
	main(args)