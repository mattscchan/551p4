import tensorflow as tf

# Example describing use of Dataset API

def parse_function(example_proto):
	# This defines what you expect the examples in your dataset to look like.
	# This is where you would do your preprocessing for example. Note that you should only use tensorflow operations. If you need to use functions from another python library, there is another way to do it.
	features = {
		'text': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)
	}

	# Calling tf.parse_single_example(example_proto, features) -- will return a dict containing a mapping between the key-value pairs of the tfrecord
	parsed_feature = tf.parse_single_example(example_proto, features)

	# This transforms our labels to one hot encoding
	one_hot_label = tf.one_hot(parsed_feature['label'])
	# THis returns the value of the field 'text' in our examples from the .tfrecords
	text_tensor = parsed_feature['text']
	# Returns these values.
	return text_tensor, one_hot_label

def create_dataset(filenames, parse_function, num_parallel_calls=1, batch_size=50, shuffle_buffer=10000, num_epochs=-1):
	# filenames: a placeholder for a list of files that you use to create a TFRecordDataset
	# parse_function: the name of a parse function that you will use to "parse" the data you get from your .tfrecords file
	# num_parallel_calls: specify an int to allow multithreading
	# batch_size: the number of elements you want returned every time you call sess.run(next_element) -- see below
	# shuffle_buffer: not that important but it is the number of elements that will be in the "buffer" when shuffled - useful so you don't have to load the whole dataset in memory for randomization
	# num_epochs: the number of times the iterator will iterate over the files that you provided. if you specify -1, it will just loop infinitely.
	# There are other transformations that you can apply to a Dataset - notably you can use a dataset.filter() function that works kind of like the data.map() function which filters out certain examples. 

	dataset = tf.data.TFRecordDataset(filenames)

	# The way the .map() function works is that is will apply some parse function to every example in the dataset. This is useful for making transformations to your entire dataset. Specify a high number of parallel calls to avoid bottlenecking your training with the preprocessing.
	dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)

	# The number of times you allow your dataset to repeat
	if num_epochs < 0:
		dataset = dataset.repeat()
	else:
		dataset = dataset.repeat(num_epochs)

	# Enable shuffling with the given buffer size
	dataset = dataset.shuffle(shuffle_buffer)

	# Specify the batch size that will be returned by the iterator
	dataset = dataset.batch(batch_size)
	# This is the iterator that will allow you to get batches
	iterator = dataset.make_initializable_iterator()
	# calling sess.run(next_element) will make the iterator return you a batch from this Dataset with the properties you gave it.
	next_element = iterator.get_next()

	return next_element, iterator


def main():
	# some number of epochs
	NUM_EPOCHS = 1000

	# This is a placeholder that will get it's value from using the feed_dict mechanism in the Session
	filenames = tf.placeholder(tf.string, shape=[None])

	# This is just a call for the helper function to create a "dataset" - also note that im not CALLING parse_function, just assigning its signature. This is because we're not evaluating, we just want to use it to build the computation graph.
	next_element, iterator = create_dataset(filenames, parse_function=parse_function, num_epochs=NUM_EPOCHS)

	with tf.Session() as sess:
		# This initializes the iterator that will have all the properties that we defined when calling create_dataset() and this is also when we assign a value to the filenames placeholder - you can actually pass many files if you want
		sess.run(iterator.initalizer, feed_dict={filenames: ['train.tfrecords']})


		# Every time you call sess.run(next_element) - you get the next batch that is generated from the iterator you defined in the create_dataset() function
		for i in range(NUM_EPOCHS):
			batch = sess.run(next_element)