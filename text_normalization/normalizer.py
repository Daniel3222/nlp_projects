import sys
import spacy
import pandas as pd
import re
from collections import Counter
import operator
import csv



def pre_normalizer(string):
    # this function takes a sentence string as input and returns the normalized sentence string. the pre-normalizer
    # is useful to treat emojis, and strong exclamations and interrogations before the tokenization, which usually
    # separates the punctuation

    string_mod = re.sub(r'(!!+)', ' STRONG_EXCLAMATION ', string)
    string_mod = re.sub(r'(\.\.+)', ' DOTS ', string_mod)
    string_mod = re.sub(r'(\?\?+)', ' STRONG_INTERROGATION ', string_mod)

    string_mod = re.sub(r':‑\)| :\)|:-\]|:\]|:-3|:3|:->|:>|8-\)|8\)|:-}|:}|:o\)|:c\)|:\^\)|=\]|=\)', ' HAPPY ',
                        string_mod)
    string_mod = re.sub(r':‑D|:D|8‑D|8D|x‑D|xD|X‑D|XD|=D|=3|B\^D', ' LAUGH ', string_mod)
    string_mod = re.sub(r':‑\(| :\( | :‑c | :c', ' SAD ', string_mod)
    string_mod = re.sub(r':\'‑\( | :\'\( ', ' CRYING ', string_mod)
    string_mod = re.sub(r';\)| ;\)', ' WINK ', string_mod)
    string_mod = re.sub(r':‑O | :O | :‑o | :o | :-0 | 8‑0 | >:O', ' SURPRISE ', string_mod)
    string_mod = re.sub(r' :P | :‑P | X‑P | XP | x‑p | xp | :‑p | :p | :‑Þ | :Þ | :‑þ | :þ | :‑b | :b | =p', ' CHEEKY ',
                        string_mod)
    string_mod = re.sub(r'( +)', ' ', string_mod)

    return string_mod


def normalizer(string):
    # this function takes a token string as input and returns the normalized string, which meets a suite of regex rules

    string_mod = re.sub(r'\d{1,2}:\d{2}:?\d?\d?[a,p]?[m]?', 'TIME', string)
    string_mod = re.sub(r'\d{1,2}\/\d{1,2}\/?\d?\d?\d?\d?', 'DATE', string_mod)
    string_mod = re.sub(r'\b(?:a*(?:h+a+)+h?)\b', 'haha', string_mod)
    string_mod = re.sub(r'\b(?:e*(?:h+e+)+h?)\b', 'haha', string_mod)
    string_mod = re.sub(r'\b(?:i*(?:h+i+)+h?)\b', 'haha', string_mod)
    string_mod = re.sub(r'\b(?:o*(?:h+o+)+h?)\b', 'haha', string_mod)
    string_mod = re.sub(r'\b((?:l+o+)+l+)\b', 'lol',  string_mod)
    string_mod = re.sub(r'[\w\.-]+@[\w\.-]+', 'EMAIL', string_mod)
    string_mod = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,3})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', 'URL', string_mod)
    string_mod = re.sub(r'\bnt\b|\bn\'t\b', 'not', string_mod)  # contractions dont on est certain
    string_mod = re.sub(r'(.)\1+', r'\1\1', string_mod)  # deux conséc pour pas pénaliser les mots comme cool, naan, etc

    return string_mod

def normalize_corpus(path):
    # globals
    blog = 'blog'
    classe = 'class'

    # init the spacy pipeline
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'vectors', 'ner', 'tagger', 'textcat'])

    df = pd.read_csv(path, names=[blog,classe])
    print(df.shape)

    # normalisation ligne par ligne de l'échantillon en utilisant la fonction normalizer()
    for index, row in df.iterrows():
        print(index)
        sent = pre_normalizer(row[blog])
        sent = nlp(sent)
        normalized_list = [(normalizer(tok.text)) for tok in sent]
        normalized_sent = ' '.join(normalized_list)
        df.iloc[index,0] = normalized_sent

    return df
