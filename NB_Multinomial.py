# from function import *
# from sklearn.feature_extraction.text import TfidfVectorizer
# import sklearn.naive_bayes
#
# path_data_1 = r"pre_data\data_1.pkl"
# path_data_2 = r"pre_data\data_2.pkl"
# path_data_3 = r"pre_data\data_3.pkl"
# path_data_4 = r"pre_data\data_4.pkl"
# path_data_5 = r"pre_data\data_5.pkl"
#
# path_label_1 = r"pre_data\label_data_1.pkl"
# path_label_2 = r"pre_data\label_data_2.pkl"
# path_label_3 = r"pre_data\label_data_3.pkl"
# path_label_4 = r"pre_data\label_data_4.pkl"
# path_label_5 = r"pre_data\label_data_5.pkl"
#
#
# def vectorTransform(X_data, X_test=None):
#     # ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
#     f = open(r'gheptu.txt', "r", encoding="utf8")
#     stopwords = f.read()
#
#     list_stop_word = stopwords.split('\n')
#
#     tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words=list_stop_word)
#     tfidf_vect_ngram.fit(X_data)
#     X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
#
#     X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
#     return X_data_tfidf_ngram, X_test_tfidf_ngram
#
#
# def train_model(classifier, X_data_tfidf, y_data, X_test_tfidf, y_test):
#     classifier.fit(X_data_tfidf, y_data)
#
#     train_predict = classifier.predict(X_data_tfidf)
#     test_predict = classifier.predict(X_test_tfidf)
#
#     print("Validation accuracy: ", sklearn.metrics.accuracy_score(train_predict, y_data))
#     print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predict, y_test))
#     print("Predicate :", test_predict)
#
#
# X_data = loadData([path_data_1,path_data_2,path_data_3,path_data_4])
# y_data = loadData([path_label_1,path_label_2, path_label_3, path_label_4])
#
#
# X_test = pickle.load(open(path_data_5, 'rb'))
# y_test = pickle.load(open(path_label_5, 'rb'))
#
# X_data_tfidf, X_test_tfidf = vectorTransform(X_data, X_test)
#
# model = sklearn.naive_bayes.MultinomialNB()
#
# def print_Ans_Mul():
#     print("Mô hình Naive Bayes với thuật toán Multinomial")
#     train_model(model, X_data_tfidf,y_data,X_test_tfidf,y_test)
#     print("--------------------------------")
from function import *
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.naive_bayes



path_data_1 = r"pre_data\data_1.pkl"
path_data_2 = r"pre_data\data_2.pkl"
path_data_3 = r"pre_data\data_3.pkl"
path_data_4 = r"pre_data\data_4.pkl"
path_data_5 = r"pre_data\data_5.pkl"
path_data_6 = r"pre_data\data_6.pkl"

path_label_1 = r"pre_data\label_data_1.pkl"
path_label_2 = r"pre_data\label_data_2.pkl"
path_label_3 = r"pre_data\label_data_3.pkl"
path_label_4 = r"pre_data\label_data_4.pkl"
path_label_5 = r"pre_data\label_data_5.pkl"
path_label_6 = r"pre_data\label_data_6.pkl"


X_data = loadData([path_data_1,path_data_2,path_data_3,path_data_4])
y_data = loadData([path_label_1,path_label_2, path_label_3, path_label_4])

X_test = pickle.load(open(path_data_6, 'rb'))
y_test = pickle.load(open(path_label_5, 'rb'))

def vectorTransform(X_data, X_test=None):
    f = open(r'gheptu.txt', "r", encoding="utf8")

    stopwords = f.read()
    list_stop_word = stopwords.split('\n')

    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words=list_stop_word)
    tfidf_vect_ngram.fit(X_data)

    X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
    X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
    return X_data_tfidf_ngram, X_test_tfidf_ngram


def train_model(X_data_tfidf, y_data, X_test_tfidf, y_test):
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(X_data_tfidf, y_data)

    train_predict = model.predict(X_data_tfidf)
    test_predict = model.predict(X_test_tfidf)

    print("Validation accuracy: ", sklearn.metrics.accuracy_score(train_predict, y_data))
    print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predict, y_test))
    print("Predicate :", test_predict)


def print_Ans_Mul():
    print("Mô hình Naive Bayes với thuật toán Multinomial")
    X_data_tfidf, X_test_tfidf = vectorTransform(X_data, X_test)
    train_model(X_data_tfidf,y_data,X_test_tfidf,y_test)
    print("--------------------------------")

