import argparse
import tensorflow as tf 
import numpy as np
import csv

RAND_SEED = int(0xCAFEBABE)

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_row(text, label):
	return tf.train.Example(
			features=tf.train.Features(
				feature={
					'text': _bytes_feature(text),
					'label': _int64_feature(label)
		})).SerializeToString()

def _write_tfrecords(filename, examples_list):
	writer = tf.python_io.TFRecordWriter(filename)

	for example in examples_list:
		writer.write(example)
	writer.close()

def main(args):
	examples = []
	np.random.seed(RAND_SEED)

	with open(args.filename, 'r', newline='') as f:
		csvreader = csv.reader(f)
		count=0
		for row in csvreader:
			if count==0:
				count += 1
				continue

			if row[3] == 'FAKE':
				row[3] = 0
			if row[3] == 'REAL':
				row[3] = 1
			
			examples.append(_convert_row(row[1]+row[2], row[3]))

		np.random.shuffle(examples)

		_write_tfrecords('test.tfrecords', examples[0:530])
		_write_tfrecords('validation.tfrecords', examples[530:1531])
		_write_tfrecords('train.tfrecords', examples[1531:])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help="Provide input file name.")
	args = parser.parse_args()
	main(args)