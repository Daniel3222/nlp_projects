import spacy
import logging
import string
from gensim.test.utils import datapath
from gensim import utils

logging.basicConfig(format=' % (asctime) s: % (levelname) s: % (message) s', level = logging.INFO)

class MyCorpus(object):
    def __iter__(self):
        corpus_path = datapath('C:/Users/MSI/PycharmProjects/facebook/tp3/train_posts.csv')
        for line in open(corpus_path, encoding='utf-8'):
            line_sans_punct = line.translate(str.maketrans('', '', string.punctuation))
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line_sans_punct)


import gensim.models
sentences = MyCorpus()
from gensim.test.utils import get_tmpfile
path = get_tmpfile("C:/Users/MSI/PycharmProjects/facebook/tp3/wor2vec_5.model")

model = gensim.models.Word2Vec(sentences=sentences, size=300, window=5, min_count=25, workers=4)
model.wv.save_word2vec_format("C:/Users/MSI/PycharmProjects/facebook/tp3/wor2vec_5.txt")
# Lancer ces deux commandes-là dans le terminal ubuntu, dans le bon folder
# gzip wor2vec.txt

# dans le terminal Pycharm
# python -m spacy init-model en C:/Users/MSI/PycharmProjects/facebook/tp3/spacy.wor2vec_5.model --vectors-loc C:/Users/MSI/PycharmProjects/facebook/tp3/wor2vec_5.txt.gz

with open('C:/Users/MSI/PycharmProjects/facebook/tp3/annotations_5.txt', 'w', encoding='utf8') as g:

    nlp = spacy.load('C:/Users/MSI/PycharmProjects/facebook/tp3/spacy.wor2vec_10.model/')
    n_vectors = 30000
    batchsize = 1500
    neighbors = nlp.vocab.prune_vectors(n_vectors, batchsize)

    new_dict = {}
    for key, value in sorted(neighbors.items()):
       if value[0] in new_dict:
           new_dict[value[0]].append((key, value[1]))
       else:
           new_dict[value[0]] = [(key, value[1])]

    def Sort_Tuple(tup):
        # key is set to sort using second element of
        # sublist lambda has been used
        tup.sort(key=lambda x: x[1], reverse=True)
        return tup

    for k, v in new_dict.items():
       y = Sort_Tuple(v)
       z = [i[0] for i in y]
       g.write(k+'\t'+ str(len(z)) + str(z)+'\n')
