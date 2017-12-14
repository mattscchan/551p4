import argparse
from gensim.models import KeyedVectors

# Original Hyperparameters found in the paper.
tau = 0.7
N = 15
gamma_1 = 0.2
gamma_2 = 2		# note that the authors used gamma_2 = inf for news - i assume they approx with very large num
delta = 0.5

Glove_Vecs_PATH = '../data/proc_glove_vec.txt'

def load_vectors():
	return KeyedVectors.load_word2vec_format(Glove_Vecs_PATH, binary=False)

def get_neighbors(target_w, glove_vecs):
	try:
		neighbors = glove_vecs.similar_by_word(target_w, topn=N)
	except:
		neighbors = []
	return neighbors

def main(args):
	glove_vecs = load_vectors()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Greedy algorithm for constructing adversarial examples.")
	args = parser.parse_args()
	main(args)
