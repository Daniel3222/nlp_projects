import pandas as pd
import numpy as np
import re,unicodedata
from collections import Counter
from datetime import datetime
startTime = datetime.now()

# --------------------- CLASSES ---------------------------------
# Bernoulli Naive Bayes classifier
# the tutorial that showed me how to code a Naive Bayes Bernoulli comes from this URL
# https://kenzotakahashi.github.io/naive-bayes-from-scratch-in-python.html

class BernoulliNB(object):
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        X = self._binarize_X(X)

        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        print(np.unique(y))
        print(separated)
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    def predict_log_proba(self, X):
        X = self._binarize_X(X)

        return [(np.log(self.feature_prob_) * x + np.log(1 - self.feature_prob_) * np.abs(x - 1)
                 ).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        X = self._binarize_X(X)

        return np.argmax(self.predict_log_proba(X), axis=1)

    def _binarize_X(self, X):
        return np.where(X > self.binarize, 1, 0) if self.binarize != None else X

# --------------- FONCTIONS --------------
# bag-of-word (BOW) function
# the BOW comes from https://maelfabien.github.io/machinelearning/NLP_2/#1-preprocessing-per-document-within-corpus

def generate_bow(allsentences):
    # Define the BOW matrix
    bag_vector = np.zeros((len(allsentences), len(vocab)))
    # For each sentence
    for j in range(len(allsentences)):
        # For each word within the sentence
        for w in allsentences[j]:
            # For each word within the vocabulary
            for i, word in enumerate(vocab):
                # If the word is in vocabulary, add 1 in position
                if word == w:
                    bag_vector[j, i] += 1
    return bag_vector

# pre-process was inspired from the TA's lab
# pre-process of strings
def process(df, t):
    df[t] = df[t].apply(lambda x : x.strip())
    df[t] = df[t].apply(lambda x : re.sub('\n', ' ', x))
    df[t] = df[t].apply(lambda x : re.sub('\[[^]]*\]', '', x))
    df[t] = df[t].apply(lambda x : re.sub("<.*?>", " ", x))
    df[t] = df[t].apply(lambda x : remove_non_ascii(x))
    return df

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ''.join(new_words)

# ----------------------------------------------------------------------------------
# -------------------------- Training Classifier with TRAIN data -------------------
# ----------------------------------------------------------------------------------

# List of stopwords for english
with open('english_stopwords.txt') as f:
    stopw = f.read()
    stopw = stopw.split('\n')

# dictionary that maps integers to classes' strings
integer_to_class = {0:'astro-ph', 1:'astro-ph.CO', 2 :'astro-ph.GA', 3: 'astro-ph.SR', 4: 'cond-mat.mes-hall',  5:
 'cond-mat.mtrl-sci', 6: 'cs.LG', 7: 'gr-qc', 8: 'hep-ph', 9: 'hep-th', 10: 'math.AP', 11: 'math.CO',
 12 :'physics.optics', 13 :'quant-ph', 14: 'stat.ML'}


# load data for testing and training
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# tokenize each row for the column 'Abstract', using a regexp
# removing stopwords and make sure every token is alphabetical characters
df_train = process(df_train, 'Abstract')
df_train['Abstract'] = df_train.apply(lambda row: (re.findall("[\w']+", row.Abstract.lower())), axis=1)
df_train['Abstract'] = df_train['Abstract'].apply(lambda x: [item for item in x if item not in stopw])
df_train['Abstract'] = df_train['Abstract'].apply(lambda x: [item for item in x if item.isalpha()])
print('tokenization', datetime.now() - startTime)

# create first instance of vocab starting with putting into a list, a list of all the tokens per abstract
sentences = df_train['Abstract'].tolist()
# flatten all the sentences in order to create a set of unique tokens afterwards
liste_vocab = [item for sublist in sentences for item in sublist]

# If we print this line, we see the initial size of the vocabulary
vocab = list(set(liste_vocab))
# print(len(vocab))

# get frequence of occurence per word
word_counts = Counter(liste_vocab)  # counts the number each time a word appears
# create a list that holds the words that we want to reject because they are'nt frequent enough
liste_min_frequency = list()
for k, v in word_counts.items():
    if v <= 4:
        liste_min_frequency.append(k)

# reject  words in list_min_frequency from the vocabulary
# update vocabulary that will be used for
# the bag of words when we use  the function "generate_bow"
vocab = list(set(vocab) - set(liste_min_frequency))

bow = generate_bow(sentences) # vectorized abstract
target = np.array(df_train.Category) # y_train data, that we called the target for each data point X
print('generate bag of words', datetime.now() - startTime)

# instantiate classifier with alpha hyper-parameter
clf = BernoulliNB(alpha=0.2)
# fit the data with the model
clf.fit(bow, target)
print('fitting', datetime.now() - startTime)

# ----------------------------------------------------------------------------------
# -------------------------- Predicting for TEST data ------------------------------
# ----------------------------------------------------------------------------------

# use process function to get rid of noise
df_test = process(df_test, 'Abstract')
# Tokenize and clean from non alpha and stopwords
df_test['Abstract'] = df_test.apply(lambda row: (re.findall("[\w']+", row.Abstract.lower())), axis=1)
df_test['Abstract'] = df_test['Abstract'].apply(lambda x: [item for item in x if item not in stopw])
df_test['Abstract'] = df_test['Abstract'].apply(lambda x: [item for item in x if item.isalpha()])

# create bow per utterance
sentences_test = df_test['Abstract'].tolist()
bow_test = generate_bow(sentences_test)
print('generated bow for test', datetime.now() - startTime)

# predict test data with Classifier
y_pred = clf.predict(bow_test)
print('finished with predictions',datetime.now() - startTime)

# rename classes with their strings instead of integer (see dictionary)
y_pred = [integer_to_class[k] for k in y_pred]
# put list of predicted classes into an array
y_pred = np.array(y_pred)

# ----------------------------------------------------------------------------------
# -------------------------- Preparing CSV for submission --------------------------
# ----------------------------------------------------------------------------------
# create a dataframe that will be the submission in .csv file

df = pd.DataFrame(data=y_pred, columns=["Category"])
df['Id'] = [i for i in range(len(df.Category))]
columns_titles = ["Id", "Category"]
df = df.reindex(columns=columns_titles)
df.to_csv('submission_ift_6390_danielgp.csv', index=False)
print('submission file created', datetime.now() - startTime)
