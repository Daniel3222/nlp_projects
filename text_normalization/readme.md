# Normalization

###### normalizer.py is a small python script that can help one normalize a corpus. In this scenario, I used that script for a task that consisted in finding the age class of blog users based on their writings.

###### The python script consists of many regular expressions meant to normalize strings. They were previously used on a dataframe, that is why the function returns a dataframe. Normalization is one important step in Natural Language Processing as it helps removing unneeded variation in strings. For example, all expressions of happyness by emojis can be normalized to one token. That can facilitate an algorithm in learning how to represent such an emotion. This script can also help you if you are looking for regular expressions to normalize emails, time series, laughs, etc.


###### The other script Named_entity_rec.py is python script that was made to select features for the age prediction task. One way to select features was to use spacy's named entity recognition module. One could also use the POS tagging to retrieve information about the users (for example, older users are more enclined to use more adjectives than younger ones, while the latter are more enclined to use more adverbs). For the NER, my hypothesis was that youngsters might be more enclined in naming people considering it's a typical topic at such an age. This script can help you count how many occurences of named entity recognition types (person, product, money, etc.) appear in texts. Finally, I've also added a way to visualize the frequency counts of these types of NER.
