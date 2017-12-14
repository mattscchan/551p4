import argparse
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize          
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords as StopWords

# Original Hyperparameters found in the paper.
tau = 0.7
N = 15
gamma_1 = 0.2
gamma_2 = 2		# note that the authors used gamma_2 = inf for news - i assume they approx with very large num
delta = 0.5

Glove_Vecs_PATH = '../data/proc_glove_vec.txt'

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

    _, x_test, _, y_test = train_test_split(x, y, test_size=0.20)
    return x_test, y_test

def load_vectors():
	return KeyedVectors.load_word2vec_format(Glove_Vecs_PATH, binary=False)

def get_neighbors(target_w, glove_vecs):
	try:
		neighbors = glove_vecs.similar_by_word(target_w, topn=N)
	except:
		neighbors = []
	return neighbors

def tokenize(example):
	punct = [',', '.', '"']	
	
	word_list = word_tokenize(example)
	word_list = [word.lower() for word in word_list]
	word_list = [word for word in word_list if word not in StopWords.words('english')]

def get_max(W):
	pass

def main(args):
	J_x = 0
	replaced_words = 0
	examples_list = []
	labels = []

	glove_vecs = load_vectors()

	if args.dataset == 'yelp':
		labels, examples_list = load_data_yelp()
	elif args.dataset == 'news':
		labels, examples_list = load_data_fakenews()
	elif args.dataset == 'spam':
		print("TODO: NOT YET IMPLEMENTED.")
	else:
		print("INVALID DATASET CHOICE.")
		return

	for example in examples_list:
		example_arr = tokenize(example)

		while J_x < tau and replaced_words < delta:
			W = []
			word_count = 1
			for word in example_arr:
				neighbors = get_neighbors(word, glove_vecs)

				if len(neighbors) > 2:
					for candidate in neighbors:
						# Check if we're not just replacing with the same word
						if candidate == word:
							continue
						# Replace word with candidate
						example_arr[word_count] = candidate 

						# Check constraints
						if check_constraint(example_arr):
							# If yes, add to working set (increase fraction of replaced words).
							W.append(example_arr)
							if replaced_words != word_count:
								replaced_words = word_count
				word_count += 1
			J_x = get_max(W)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Greedy algorithm for constructing adversarial examples.")
	parser.add_arguments('model', help='Choose the model that we are going to use to generate adversarial examples.')
	parser.add_arguments('dataset', help='Choose the dataset to evaluate on.')
	args = parser.parse_args()
	main(args)
