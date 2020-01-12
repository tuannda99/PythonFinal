from pyvi import ViTokenizer
import gensim
import re
import nltk
import pickle
import os

# -----------------------------Xử lý văn bản đầu vào-------------------------------------------------

EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
URL = re.compile(r"https?:\/\/(?!.*:\/\/)\S+")
PHONE = re.compile(r"(09|01[2|6|8|9])+([0-9]{8})\b")
MENTION = re.compile(r"@.+?:")
NUMBER = re.compile(r"\d+.?\d*")
DATETIME = re.compile(r'\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}')
#PROPER_NOUN = re.compile(r'(([A-Z]([a-z]+|\.+))+(\s[A-Z][a-z]+)+)|([A-Z]{2,})|([a-z][A-Z])[a-z]*[A-Z][a-z]*')
SPECIAL = re.compile(r'([^\w\s]){2}')
EMOJI = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticonss
        u"\U0001F680-\U0001F6FF"  # transport & map symb
        u"\U0001F300-\U0001F5FF"  # symbols & pictographols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

def getLabel(sentence):
    label = ''
    i = 1
    for c in sentence:
        if c == '|':
            break
        else:
            label += c
            i += 1
    return label, i


def file_processing(path):
    f = open(path, "r", encoding="utf8")
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]
    X = []
    labels = []
    lines = f.readlines()

    for sentence in lines:
        label, index = getLabel(sentence)
        sentence = sentence[index:len(sentence)]

        preprocessing_sentence = preprocessing(sentence)
        X.append(preprocessing_sentence)
        labels.append(label)

        dumpDataFileName = 'pre_data/'+base+'.pkl'
        dumpLablelFileName = 'pre_data/label_'+base+'.pkl'

        pickle.dump(X, open(dumpDataFileName, 'wb'))
        pickle.dump(labels, open(dumpLablelFileName, 'wb'))


def preprocessing(sentence):
    sentences = nltk.sent_tokenize(sentence)
    preprocessing_sentence = ''

    for sentence in sentences:
        # url regex
        sentence = re.sub(EMAIL,' EMAIL ', sentence)
        sentence = re.sub(URL,' URL ', sentence)
        sentence = re.sub(PHONE,' PHONE ', sentence)
        sentence = re.sub(EMAIL,' EMAIL ', sentence)
        sentence = re.sub(MENTION,' MENTION ', sentence)
        sentence = re.sub(NUMBER,' NUMBER ', sentence)
        sentence = re.sub(DATETIME,' DATETIME ', sentence)
        #sentence = re.sub(PROPER_NOUN,' PROPER_NOUN ', sentence)
        sentence = re.sub(EMOJI,' EMOJI ', sentence)

        sentence = gensim.utils.simple_preprocess(sentence)

        sentence = ' '.join(sentence)
        sentence = ViTokenizer.tokenize(sentence)
        preprocessing_sentence += ''.join(sentence) + "."
    return preprocessing_sentence


def dumpData():
    path_1 = r'data\data_1.txt'
    path_2 = r'data\data_2.txt'
    path_3 = r'data\data_3.txt'
    path_4 = r'data\data_4.txt'
    path_5 = r'data\data_5.txt'
    path_6 = r'data\data_6.txt'

    list = [path_1, path_2, path_3, path_4, path_5, path_6]

    for path_ in list:
        file_processing(path_)

def loadData(list):
    X_data = []
    for path in list:
        X_data += pickle.load(open(path, 'rb'))
    return X_data

dumpData()
