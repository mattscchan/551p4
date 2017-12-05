import argparse
from nltk.model.ngram import NGramModel

# Original Hyperparameters found in the paper.
tau = 0.7
N = 15
gamma_1 = 0.2
gamma_2 = 2		# note that the authors used gamma_2 = inf for news - i assume they approx with very large num
delta = 0.5

class MLENgramModel(NGramModel):

	def __init__(self, ngram, text, pad_left=False, pad_right=False, smoothing=None):
		NGramModel.__init__(self, ngram, text, pad_left, pad_right, smoothing)

	def score(self, context, word):
		# how many times word occurs with context
		ngram_count = self.ngrams[context][word]
		# how many times the context itself occurred we take advantage of
		# the fact that self.ngram[context] is a FreqDist and has a method
		# FreqDist.N() which counts all the samples in it.
		context_count = self.ngram[context].N()

		# In case context_count is 0 we shouldn't be dividing by it 
		# and just return 0
		if context_count == 0:
		    return 0
		# otherwise we can return the standard MLE score
		return ngram_count / context_count

# def calculate_language_model(input, stopword=False, )

def main(args):
	model = MLENgramModel(2, "Hi my name is Matt.")
	print(score("How are you Matt on this fine day?"))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Greedy algorithm for constructing adversarial examples.")
	args = parser.parse_args()
	main(args)
