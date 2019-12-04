from collections import Counter
import pandas as pd
import re as regex
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
import gensim
import nltk
import spacy
import seaborn as sns

nlp = spacy.load("en_core_web_sm", disable=['parser', 'vectors', 'textcat'])

in_f = 'normalize.out'
out_f = 'ner.out'
size = 10000


def preprocess_text(in_f, out_f, size):
    write_header = True
    reader = pd.read_table(in_f, names=['blog', 'classe'], chunksize=size, delimiter=',')
    for chunk in reader:
        for index, row in chunk.iterrows():
            sent = nlp(row[0])
            nb_label = []
            for ent in sent.ents:
                nb_label.append(ent.label_)
            d = Counter(nb_label)
            chunk.loc[index, 'PERSON'] = d.get("PERSON", 0)
            chunk.loc[index, 'PRODUCT'] = d.get("PRODUCT", 0)
            chunk.loc[index, 'MONEY'] = d.get("MONEY", 0)
            chunk.loc[index, 'GPE'] = d.get("GPE", 0)
            chunk.loc[index, 'EVENT'] = d.get("EVENT", 0)
        print(chunk)
        chunk.to_csv(out_f, header=write_header, mode='a')
        write_header = False


preprocess_text(in_f, out_f, size)

###### Visualize  #######

path_read = 'ner.out'
df = pd.read_csv(path_read)
df['sum'] = df.iloc[:,3:].sum(axis=1)

df.iloc[:,3:8] = df.iloc[:,3:8].div(df['sum'], axis=0)
df = df.fillna(value=0)


summary = df.groupby(['classe'], as_index=False).mean()
summary = summary[['classe', 'PERSON', 'PRODUCT', 'MONEY', 'GPE', 'EVENT']]
summary = pd.melt(summary, id_vars='classe', var_name='NER', value_name='Fréquence normalisée')

sns.factorplot(x='classe', y='Fréquence normalisée', hue='NER', data=summary, kind='bar', palette='colorblind')
