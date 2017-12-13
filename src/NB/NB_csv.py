from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv


# =================
# NAIVE BAYES MODEL
# =================


def model(x_train, y_train, x_test, y_test):
    clf = MultinomialNB()
    print("Training Naive Bayes model...")
    model = clf.fit(x_train, y_train)

    print("Testing Naive Bayes model...")
    predicted = model.predict(x_test)

    return accuracy_score(predicted, y_test)

# ===========================
# BAG-OF-WORDS REPRESENTATION / SHOULD
# ===========================


def bag_of_words(x_train, x_test):
    print("Converting to Bag of Words representation...")
    vectorizer = CountVectorizer(stop_words=None)
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    return x_train_bow, x_test_bow

# =========================================
# LOADING DATA / DIFFERENT FOR EACH DATASET
# =========================================


def load_data_yelp(filename, x, y):
    print("Loading Yelp data ...")
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x.append(row[1])
            y.append(row[0])
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

    return x_train, y_train, x_test, y_test

# ==========
# MAIN
# ==========


def main():

    # ==============
    # YELP
    # ==============

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_yelp = "../csv/yelp_dataset/train.csv"

    test_yelp = "../csv/yelp_dataset/test.csv"

    x_train, y_train = load_data_yelp(train_yelp, x_train, y_train)

    x_test, y_test = load_data_yelp(test_yelp, x_test, y_test)

    x_train, x_test = bag_of_words(x_train, x_test)

    print(model(x_train, y_train, x_test, y_test))


    # =============
    # FAKE NEWS
    # =============

    news_data = "../csv/fakenews_dataset/fake_news.csv"

    x_train, y_train, x_test, y_test = load_data_fakenews(news_data)

    x_train, x_test = bag_of_words(x_train, x_test)

    print(model(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    main()
