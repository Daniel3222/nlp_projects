# imports
import json
import re
from functools import partial
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Ouvre et répare les messages de facebook provenant du fichier json fourni par Facebook
fix_mojibake_escapes = partial(re.compile(rb'\\u00([\da-f]{2})').sub, lambda m: bytes.fromhex(m.group(1).decode()))

# Open facebook messages in json format
with open('message_1.json', 'rb') as binary_data:
    repaired = fix_mojibake_escapes(binary_data.read())
data = json.loads(repaired.decode('utf8'))
# Parcourir le fichier jusqu'à la section message
messages = data.get('messages')

liste_sender = []
liste_message = {}
for all_messages in messages:
    sender = all_messages.get('sender_name')
    message = all_messages.get('content')
    if message is None:
        message = '1'
    if sender not in liste_message:
        liste_message[sender] = []
    liste_sender.append(sender)
    liste_message[sender].append(message)


nlp = spacy.load('fr_core_news_md')
for x in liste_message:
    texte_tokenise = liste_message[x]
    texte_detok = TreebankWordDetokenizer().detokenize(texte_tokenise)
    doc = nlp(texte_detok.lower())

    # Faire une fonction d'expression régulière pour retirer les tokens à 2 strings, retirer les j devant les verbes,
    # lemmatiser tous les mots pour pouvoir faire du clustering après.
    customize_stop_words = [
        "d’", "qu’", "d'", "qu'", "j'", "j’", "c'", "c’", "l'", "l’", "t'", "t’", "-ce", "n'", "n’", "lol", "pis", 'pi',
        'ca']
    liste_tokens = []
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True
    for token in doc:
        if not token.is_punct | token.is_stop:
            liste_tokens.append(token.text)
    word_freq = Counter(liste_tokens)
    common_words = word_freq.most_common(10)
    # you can now print the 10 most common words in a conversation
    print('Pour ' + x, ',les mots plus communs sont : ' + str(common_words))

messages_count = liste_message.values()
total_message_envoyes = []
for sublist in messages_count:
    for item in sublist:
        total_message_envoyes.append(item)
# You could also count how many messags have been sent on the conversation
print('Le nombre total de messages envoyés sur la convo de la boys = ' + str(len(total_message_envoyes)) + '\n')


# Statistiques
# In this section we will check at some of the data in terms of basic statistics
# Longest message per person, average of how many characters are sent per message
# Total number of characters sent
moyenne = {}
somme = {}
maximum = {}
for person in liste_message:
    taille_message = []
    for texte in liste_message[person]:
        taille_message.append(len(texte))
    moyenne[person] = sum(taille_message)/len(taille_message)
    somme[person] = sum(taille_message)
    maximum[person] = max(taille_message)
    print('=='+person+'==' + '\n', '- moyenne de caractères par message : ' + str(round(moyenne[person])) + '\n',
          '- total des caractères écrits : ' + str(somme[person]) + '\n',
          '- plus long message envoyé : ' + str(maximum[person]) + '\n')


# You can also check the reactions data
# For example how many time one person reacted in total to others' messages
liste_reactions = []
for all_messages in messages:
    reactions = all_messages.get('reactions')
    if type(reactions) == list:
        liste_reactions.append(reactions)

flat_list = []
for sublist in liste_reactions:
    for item in sublist:
        flat_list.append(item)

liste_reactionneur = []
for x in flat_list:
    actor = x.get('actor')
    liste_reactionneur.append(actor)

reactionneur_freq = Counter(liste_reactionneur)
ordre_reactionneurs = reactionneur_freq.most_common(10)
liste_plot_nom = []
liste_plot_valeur = []
print('\n', ordre_reactionneurs)


# One can also put in a plot the results of this search on reactions
for tuple_x in ordre_reactionneurs:
    prenom = tuple_x[0].split()
    liste_plot_nom.append(prenom[0])
    liste_plot_valeur.append(tuple_x[1])

plt.title('')
plt.plot(liste_plot_nom, liste_plot_valeur, '*')
plt.xticks(liste_plot_nom, rotation='25')
plt.ylabel('Nombre total de réactions à un message')
plt.show()

# -------------------------------------
# Script pour les timestamps de chacun
# ------------------------------------

# Fonction Permettant d'arrondir les heures
def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + timedelta(hours=t.minute//30))


list_ts = []
for all_messages in messages:
    timestamp = all_messages.get('timestamp_ms')
    sender = all_messages.get('sender_name')
    timestamp_sender = (sender, float(timestamp))
    list_ts.append(timestamp_sender)

dictionary = {}
for x in list_ts:
    if x[0] not in dictionary:
        dictionary[x[0]] = []  # ici on définit clé-valeur, je précise que la clé c'est l'indice 0 des tuples
    time_boys = hour_rounder(datetime.fromtimestamp(x[1]/1000)).strftime('%H')
    dictionary[x[0]].append(time_boys)

hours = {}
totaux = {}
for x in dictionary:
    hours_frequence = Counter(dictionary[x])
    common_hours = sorted(hours_frequence.most_common(24), key=lambda z: z[0])
    integ = []
    total = []
    for tup in common_hours:
        integ.append(int(tup[0]))
        total.append(int(tup[1]))
    hours[x] = integ
    totaux[x] = total
    plt.plot(hours[x], totaux[x])
    plt.title(x)
    plt.ylabel('Quantité de messages')
    plt.xlabel('heure de la journée')
    plt.show()
