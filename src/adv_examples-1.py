import argparse
import math
import os
import re
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize          
from nltk.corpus import stopwords as StopWords
from collections import defaultdict
from nltk import trigrams
from tensorflow.python.keras.models import load_model

# Original Hyperparameters found in the paper.
tau = 0.7
N = 15
gamma_1 = 0.2
gamma_2 = 2		# note that the authors used gamma_2 = inf for news - i assume they approx with very large num
delta = 0.5

Glove_Vecs_PATH = '../data/proc_glove_vec.txt'
Glove_Vecs_PATH = os.path.join(dirname(dirname(os.getcwd())), 'data/glovevec/proc_glove_vec.txt')

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    return x_test, y_test, x_train, y_train

def load_vectors():
    return KeyedVectors.load_word2vec_format(Glove_Vecs_PATH, binary=False)

def get_neighbors(target_w, glove_vecs):
    neighbors = []
    try:
        neighbors = glove_vecs.similar_by_word(target_w, topn=N)
        neighbors = [el[0] for el in neighbors]	# just get the word, don't care bout similarity
    except:
        pass
    return neighbors

def tokenize(example):
    punct = [',', '.', '"', '\n']	
    
    word_list = word_tokenize(example)
    word_list = [re.sub('[^a-zA-Z0-9\s]', '', word.lower()) for word in word_list if word not in punct]
    return word_list

def get_max(W, classifier):
    # Assumes W is a zip of sentences and labels
    def test(model, x, y):
        result = model.evaluate(x, y, verbose=0)
        accuracy = result[1]

    worst_acc = 1.0
    sentence = []
    for w in W:
        if test(classifier, w[0], w[1]) < worst_acc:
            sentence = w[0]
    return sentence

def create_trigram_model(x_train, examples):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    # Training
    for _, s in enumerate(x_train):
        w1 = None
        w2 = None
        for i, w in enumerate(tokenize(s)):
            model[(w1, w2)][w] += 1
            w1 = w2
            w2 = w
        model[(w2, w)][None] += 1
        model[(w, None)][None] += 1

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model

# Prob of sentence
def check_trigram(model, example):
    for _, s in enumerate(example):
        w1 = None
        w2 = None
        for i, w in enumerate(tokenize(s)):
            count += model[w1, w2][w]
            w1 = w2
            w2 = w
        count += model[w1, w][None]
        count += model[w, None][None]
    return loat(count)/(i+1)

# Difference of log of prob of sentences
def check_constraints(trigrams, condition, sentence_ori, sentence_new):
    prob_ori = check_trigrams(trigrams, sentence_ori)
    prob_new = check_trigrams(trigrams, sentence_new)
    return abs(math.log(prob_new) - math.log(prob_new)) < condition

def main(args):
    classifer_path = args.model
    J_x = 0
    replaced_words = 0
    examples_list = []
    labels = []
    x_train = []
    y_train = []
    classifier = load_model(classifier_path)
    syntactic_bound = 2
    delta = delta

    glove_vecs = load_vectors()

    if args.dataset == 'yelp':
        train_yelp = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/yelp_dataset/train.csv')
        test_yelp = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/yelp_dataset/test.csv')

        examples_list, labels = load_data_yelp(train_yelp, examples_list, labels)
        x_train, y_train = load_data_yelp(test_yelp, x_train, y_train)

    elif args.dataset == 'news':
        news_data = os.path.join(dirname(dirname(os.getcwd())), 'data/csv/fakenews_dataset/fake_news.csv')

        examples_list, labels, x_train, y_train = load_data_fakenews(news_data)
    elif args.dataset == 'spam':
        print("TODO: NOT YET IMPLEMENTED.")
    else:
        print("INVALID DATASET CHOICE.")
        return

    trigram_model = create_trigram_model(x_train, examples_list)

    modified_samples = 0
    for example in examples_list:
        example_arr = tokenize(example)
        ori_arr = example_arr

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
                        if check_constraint(trigram_model, syntactic_bound, kexample_arr, ori_arr):
                            # If yes, add to working set (increase fraction of replaced words).
                            W.append(example_arr)
                            if replaced_words != word_count:
                                replaced_words = word_count
                word_count += 1
            J_x = get_max(W, classifier)
        modified_samples += 1

    print(modified_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Greedy algorithm for constructing adversarial examples.")
    parser.add_arguments('model', help='Choose the model that we are going to use to generate adversarial examples.')
    parser.add_arguments('dataset', help='Choose the dataset to evaluate on.')
    args = parser.parse_args()
    main(args)
