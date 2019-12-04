import json
import re
from functools import partial

# expression régulière permettant de régler le problème d'encodage provenant de Facebook
# (leur encodage de base a de la misère à lire les accents)
fix_mojibake_escapes = partial(re.compile(rb'\\u00([\da-f]{2})').sub, lambda m: bytes.fromhex(m.group(1).decode()))

# Ouvre le fichier json de facebook, applique la fonction du haut ce qui répare l'encodage
with open('message_1.json', 'rb') as binary_data:
    repaired = fix_mojibake_escapes(binary_data.read())
data = json.loads(repaired.decode('utf8'))

# ouvrir un fichier qui s'appellera "message_de_facebook" et ajouter l'information qu'on veut dedans.
myfile = open('messages_de_facebook', "w", encoding="utf8")
messages = data.get('messages')
for all_messages in messages:
    content = all_messages.get('content')
    try:
        myfile.writelines(content + '\n')
    except:
        myfile.writelines('None ')
myfile.close()