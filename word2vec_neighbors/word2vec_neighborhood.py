import spacy
import logging
import string
import gensim.models
from gensim.test.utils import datapath
from gensim import utils
from gensim.test.utils import get_tmpfile

# Logs info on the process
logging.basicConfig(format=' % (asctime) s: % (levelname) s: % (message) s', level = logging.INFO)

# Define the Class that will help you create the Word2Vec model
# it also reads the document line per line, so that it is memory efficient
class MyCorpus(object):
    def __iter__(self):
        corpus_path = datapath('your_path')
        for line in open(corpus_path, encoding='utf-8'):
            line_sans_punct = line.translate(str.maketrans('', '', string.punctuation))
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line_sans_punct)
            
sentences = MyCorpus()
path = get_tmpfile("C:/Users/MSI/PycharmProjects/facebook/tp3/wor2vec_5.model")

# 4 parameters can be tune
# size is the number of dimensions that your embedding will have
# window is the number of neighboors taken into account (context)
# workers are the number of CPU you want to use
model = gensim.models.Word2Vec(sentences=sentences, size=300, window=5, min_count=25, workers=4)
model.wv.save_word2vec_format("C:/Users/MSI/PycharmProjects/facebook/tp3/wor2vec_5.txt")

# Afterwards, to be used in SpaCy, one needs to make it readible by SpaCy.
# To do this you first have to zip it : gzip word2vec.txt

# And then pass this commmand :
# python -m spacy init-model en word2vec.model --vectors-loc word2vec.txt.gz


# You can open a txt file where the neighbors will be put in
with open('a_file', 'w', encoding='utf8') as g:

    nlp = spacy.load('word2vec.model')
    n_vectors = 30000
    batchsize = 1500
    neighbors = nlp.vocab.prune_vectors(n_vectors, batchsize)
    
    # The neighbors are calculated with prune_vectors
    # Which is a module that will map similar words on the same vector so that the
    # resulting model is pruned
    
    new_dict = {}
    for key, value in sorted(neighbors.items()):
       if value[0] in new_dict:
           new_dict[value[0]].append((key, value[1]))
       else:
           new_dict[value[0]] = [(key, value[1])]

    def Sort_Tuple(tup):
        tup.sort(key=lambda x: x[1], reverse=True)
        return tup

    # here we create an easy to read format to see
    # The selected words in the model and which words have been mapped onto the key word
    for k, v in new_dict.items():
       y = Sort_Tuple(v)
       z = [i[0] for i in y]
       g.write(k+'\t'+ str(len(z)) + str(z)+'\n')
