# Imports and modules from NLTK
import nltk
import numpy as np
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Functions
def generate_bow(allsentences):
        # We will delimeter the bow array (or matrix)
        bag_vector = np.zeros((len(allsentences), len(vocab)))
        # For each sentence
        for j in range(len(allsentences)):
            # For each word within the sentence
            for w in allsentences[j]:
                # For each word within the vocab
                for i,word in enumerate(vocab):
                    # If the word is in vocabulary, ADD 1
                    if word == w: 
                        bag_vector[j,i] += 1
        return bag_vector


stopwords_english = stopwords.words('english')
stemmer_english = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')

# create liste of tuples [('text', target), ... ]
# Then tokenize the 'text' part
tuples_reviews = open_and_split('C:/Users/MSI/Downloads/train.txt')  
tokenized_text = [tokenizer.tokenize(review[0]) for review in tuples_reviews]


# Filter sentences using the stopwords list
not_stopwords = {'why', 'how', 'no', 'nor', 'not','ain', 't'} 
stopwords_english = set([word for word in stopwords_english if word not in not_stopwords])
filtered_sentences = []
for sent in tokenized_text:
    filtered_sent = [w for w in sent if not w in stopwords_english]
    filtered_sentences.append(filtered_sent)

# Stemming of the filtered sentences
stemmed_sentences = [] # instanciate list where all the sentences will be
for sent in filtered_sentences:
    stemmed_words = []
    for words in sent:
        stemmed_mot = stemmer_english.stem(words)
        stemmed_words.append(stemmed_mot)
    stemmed_sentences.append(stemmed_words)

# This list contains solely the vocabulary that has been stemmed
vocab = list(set(stemmed_words_frequency))

array_target = [ int(review[1]) for review in tuples_reviews] # list of all the targets for each sentence
target = np.array(array_target) # transform list into array

# We now generate the bag of word model
bow = generate_bow(stemmed_sentences)

# Now the training part
# We had imported MultinomialNB previously from sklearn
# We are choosing Multinomial because between the three types of NB classifier,
# it is the one that fits most. Bernouilli is for only binary (0 and 1), Gaussian 
# is for floats. We have numbers that vary between 0 and more, but only integers.
clf = MultinomialNB()

# We then fit our model using our Bag-of-Word model and the targets
clf.fit(bow, target)


# Now we proceed to preprocess the test data
# Open and split test
tuples_reviews_test = open_and_split('C:/Users/MSI/Downloads/test.txt')

# tokenize test
tokenizer = RegexpTokenizer(r'\w+')
tokenized_test = [tokenizer.tokenize(review[0]) for review in tuples_reviews_test]

# CREATE VOCABULARY
# We flattened the list to have alltogheter all tokens in one list
flat_list_test = [item for sublist in tokenized_test for item in sublist]
filtered_sentence_test = [w for w in flat_list_test if not w in stopwords_english]
# This list will help us create the vocabulary
stemmed_words_frequency_test = []
for word in filtered_sentence_test:
    word_stem_test = stemmer_english.stem(word)
    stemmed_words_frequency_test.append(word_stem_test)

# This is a list of all sentences, they will be tokenized and filtered
filtered_sentences_test = []
for sent in tokenized_test:
    filtered_sent_test = [w for w in sent if not w in stopwords_english]
    filtered_sentences_test.append(filtered_sent_test)

# We take the filtered sentences, which are lists of lists
# And we stem them
stemmed_sentences_test = []
for sent in filtered_sentences_test:
    stemmed_words_test = []
    for words in sent:
        stemmed_mot_test = stemmer_english.stem(words)
        stemmed_words_test.append(stemmed_mot_test)
    stemmed_sentences_test.append(stemmed_words_test)

# So here we first convert our list to a set, because we only want the
# vocabulary, so each individual word one time.
# We then convert back to a list that contains each word, once
vocab_test = list(set(stemmed_words_frequency_test))

array_target_test = [int(review[1]) for review in tuples_reviews_test]
target_test = np.array(array_target_test)

# We now generate the bag of words
bow_test = generate_bow(stemmed_sentences_test)

y_pred = clf.predict(bow_test)
y_true = target_test

print('accuracy score : ',accuracy_score(y_true, y_pred))
