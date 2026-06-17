from data_loading import load_data
import pandas as pd
import nltk
from collections import Counter
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 200)
df0 = load_data()

# print((df0['statement'] == '#NAME?').sum()) # TODO: kaputte Zeilen, muss bereinigt werden

# Normalisieren des Textes
df0['statement'] = df0['statement'].str.lower()

# Nur die Depression-Statements herausfiltern
depression_text = df0[df0['status'] == 'Depression']['statement']
# print(depression_text.head())
# print(depression_text.describe())

# Alle Statements zu einem großen String verbinden, durch Leerzeichen getrennt
grosser_text = depression_text.str.cat(sep=' ')
# grosser_text = grosser_text.lower() # TODO: mit Team klären – lower() zentral in cleaning oder lokal?

# Den String in einzeilne Wörter zerlegen (an Leerzeichen)
woerter = grosser_text.split()

# Häufigkeiten zählen und die Top 20 ausgeben
zaehler1= Counter(woerter)
# print(zaehler1.most_common(20))

# Häufigkeiten zählen und die Top 20 ausgeben ohne Stoppwörter aus NLTK
woerter_ohne_stopwords = [wort for wort in woerter if wort not in stop_words]
zaehler2 = Counter(woerter_ohne_stopwords)
for wort, anzahl in zaehler2.most_common(20): # tuple unpacking
    print(f"{anzahl:>10}  {wort}")