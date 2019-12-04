import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from collections import Counter
import numpy as np

# Define functions
def open_and_split(data_path):
  with open(data_path) as f:
    reviews = f.read().lower()
    reviews = reviews.split('\n')
    tuples_reviews = [tuple(s.split('\t')) for s in reviews]
    return tuples_reviews

def generate_bow(allsentences):
  # define the bag-of-words array
  bag_vector = np.zeros((len(allsentences), len(vocab)))
  # for each sentence
  for j in range(len(allsentences)):
      # for each word within the sentence
      for w in allsentences[j]:
          # for each word within the vocab
          for i, word in enumerate(vocab):
              # if the word is in vocab add 1 in array
              if word == w:
                  bag_vector[j, i] += 1
  return bag_vector

# Open data and split
tuples_reviews = open_and_split(data_path)

# tokenize
tokenizer = RegexpTokenizer(r'\w+')
tokenized_text = [tokenizer.tokenize(review[0]) for review in tuples_reviews]

# Create a unique list that contains all the tokens
flat_list = [item for sublist in tokenized_text for item in sublist]

# remove stopwords, and filter
stopwords_english = stopwords.words('english')
not_stopwords = {'why', 'how', 'no', 'nor', 'not','ain', 't'}
stopwords_english = set([word for word in stopwords_english if word not in not_stopwords])
filtered_sentence = [w for w in flat_list if not w in stopwords_english]

# Stemming
stemmer_english = SnowballStemmer('english')
stemmed_words_frequency = []
for word in filtered_sentence:
  word_stem = stemmer_english.stem(word)
  stemmed_words_frequency.append(word_stem)

fdist = FreqDist(stemmed_words_frequency)
# Get the frequency of the 2000 most common unigrams
# We chose 2000, because it's the best combined, it is possible (not likely)
# that in the end, only unigrams will be considered best features
most_2000_unigrams = fdist.most_common(2000)

# Now we will proceed to the same steps, but for the bigrams

# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = ngrams(data, num)
    return [' '.join(grams) for grams in n_grams]

# We repeat all the previous operations

stopwords_english = stopwords.words('english')
stemmer_english = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')

# create liste of tuples [('text', target), ... ]
# Then tokenize the 'text' part
tuples_reviews = open_and_split('train.txt')
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


# This is going to be the list of lists of bigrams [ [bigrams], [bigrams], ... ]
bigrams_list = []
for s in stemmed_sentences:
    bigrams = extract_ngrams(s, 2)
    bigrams_list.append(bigrams)

# in one list, get all the bigrams from the whole data
flat_list_bigram = [item for sublist in bigrams_list for item in sublist]

fdist_bi = FreqDist(flat_list_bigram)
most_2000_bigrams = fdist_bi.most_common(2000)

# We will now create two dictionaries so that we can merge them easily
dictio_bigram = {}
for tup in most_2000_bigrams:
  dictio = {tuple(tup[0].split(" ")) : tup[1]}
  dictio_bigram.update(dictio)

dictio_unigram = dict(most_2000_unigrams)

merged_dict_1_2_grams = {}
merged_dict_1_2_grams = Counter(dictio_unigram) + Counter(dictio_bigram)

# Now that we have merged our unigrams and bigrams,
# We can proceed to find the features that are the most frequent combined

most_frequent_combined = FreqDist(merged_dict_1_2_grams).most_common(2000)

# We will now extract them separetaly, because we want to compare unigrams to
#  unigrams and bigrams to bigrams when we will be training
#  so we create two separate lists, that combined, will contain the 2000 most
# common features.
liste_uni = []
liste_bi = []

for x in most_frequent_combined:
  if type(x[0]) == tuple:
    liste_bi.append(x[0])
  else:
    liste_uni.append(x[0])


liste_of_splitted_bigrams = []
for sent in bigrams_list:
  for bigr in sent:
      split_bigram = tuple(bigr.split(" "))
      liste_of_splitted_bigrams.append(split_bigram)

vocab = liste_uni + liste_bi
# combined_sentences = stemmed_sentences + liste_of_splitted_bigrams
uni_bow = generate_bow(stemmed_sentences)
bi_bow = generate_bow(bigrams_list)

# We now horizontally stack the two arrays, because they can't be directly
# combined. This is an efficient way considering they don't possess the
# same dimensions.
combined_arrays = np.hstack((uni_bow, bi_bow))

# get targets, they are the same since the beginning
array_target = [int(review[1]) for review in tuples_reviews]
target = np.array(array_target)

# Now we train a classifier
clf = MultinomialNB()
clf.fit(combined_arrays, target)


# --------------------- PREPROCESS TEST DATA FOR 1-GRAM and 2-GRAM-------------
# Now we predict with the classifier
# So we repeat the operations with the test dataset
# for both unigrams and bigrams

tuples_reviews_test = open_and_split(data_path_test)

# the vocab is the same

uni_bow_test = generate_bow(stemmed_sentences_test)
bi_bow_test = generate_bow(bigrams_list_test)

# We now horizontally stack the two arrays, because they can't be directly
# combined. This is an efficient way considering they don't possess the
# same dimensions.
combined_arrays = np.hstack((uni_bow_test, bi_bow_test))

# get targets, they are the same since the beginning
array_target_test = [ int(review[1]) for review in tuples_reviews_test]
target_test = np.array(array_target_test)


y_pred = clf.predict(combined_arrays)
y_true = target_test

print(accuracy_score(y_true, y_pred))
# it yields a score of 81.4%
# Conclusion, it is way better than when we were only using bigram features
# The combination is good, but not as good as unigram features.
# The rise in accuracy might be explained by the fact, that most of the features
# in this model were of unigram type.
