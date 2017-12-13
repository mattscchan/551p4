import gensim 
import argparse

OUTPUT_FILE = '../../data/word2vec/proc_glove.txt'

def main(args):
	gensim.scripts.glove2word2vec.glove2word2vec(args.file, OUTPUT_FILE)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('file')
	args = parser.parse_args()
	main(args)