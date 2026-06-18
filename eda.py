from data_loading import load_data
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


df0 = load_data()
# if __name__ == "__main__":
#     pd.set_option('display.max_colwidth', 200)
#     pd.set_option('display.width', 200)
#     print(df0['status'].unique())
#     print(df0['status'].value_counts())

# print((df0['statement'] == '#NAME?').sum()) # TODO: kaputte Zeilen, muss bereinigt werden

# Normalisieren des Textes
df0['statement'] = df0['statement'].str.lower()

# === Häufigkeitsanalyse einer Einzelklasse (Depression) – Demonstration ===

def haeufigkeitsanalyse_depression(df0, stop_words):
    """Zeigt: rohe Worthäufigkeiten sind von Stopwörtern dominiert, daher Filterung.
    Diese Analyse ist exemplarisch; die klassenübergreifende Auswertung folgt via TF-IDF."""

    # Nur die Depression-Statements herausfiltern
    depression_text = df0[df0['status'] == 'Depression']['statement']
    # print(depression_text.head())
    # print(depression_text.describe())

    # Alle Statements zu einem großen String verbinden, durch Leerzeichen getrennt
    big_text = depression_text.str.cat(sep=' ')
    # print(big_text)
    # big_text = big_text.lower() # TODO: mit Team klären – lower() zentral in cleaning oder lokal?

    # Den String in einzelne Wörter zerlegen (an Leerzeichen)
    words = big_text.split()

    # Häufigkeiten zählen und die Top 20 ausgeben
    counter1 = Counter(words)
    # print(zaehler1.most_common(20))

    # Häufigkeiten zählen und die Top 20 ausgeben ohne Stoppwörter aus NLTK
    words_wo_stopwords = [word for word in words if word not in stop_words]
    counter2 = Counter(words_wo_stopwords)
    print("---------------------Top 20 Inhaltswörter in der Depression Klasse---------------------")
    for word, count in counter2.most_common(20):  # tuple unpacking
        print(f"{count:>10}  {word}")


def build_class_docs(df):
    """Fügt pro Klasse alle Statements zu einem großen String zusammen.
    Gibt ein Dictionary zurück: {Klassenname: Gesamttext}."""
    # Dictionary anlegen, das pro Klasse den zusammengefügten Gesamttext speichert
    # (Schlüssel = Klassenname, Wert = ein großer String mit allen Statements der Klasse)
    class_doc = {}
    for cl in df['status'].unique():
        # Alle Statements herausfiltern, die zur aktuellen Klasse gehören.
        text = df[df['status'] == cl]['statement']
        # Die einzelnen Statements zu einem großen String zusammenfügen, durch Leerzeichen getrennt
        # Den Gesamttext unter dem Klassennamen im Dictionary ablegen.
        class_doc[cl] = text.str.cat(sep=' ')
    return class_doc

# Kontroll-Ausgabe: pro Klasse einen kurzen Ausschnitt anzeigen, um den Inhalt zu prüfen
if __name__ == "__main__":
    # Funktion aufrufen und Ergebnis auffangen
    class_doc = build_class_docs(df0)
    # Kontroll-Ausgabe: pro Klasse einen kurzen Ausschnitt anzeigen
    for cl, text in class_doc.items():
        print(f"---------------------Texte in der Klasse {cl}---------------------")
        print()
        print(text[:300])
        print()

    haeufigkeitsanalyse_depression(df0, stop_words)


# Work in progress
# Ziel: pro Klasse die typischsten Wörter über TF-IDF finden

# Baut aus dem DataFrame ein Dict {Klassenname: ein großer String mit allen Texten der Klasse}
# -> 7 "Dokumente", eins pro Klasse
class_doc = build_class_docs(df0)

# TfidfVectorizer: wandelt Texte in eine TF-IDF-Zahlenmatrix um.
# stop_words='english' wirft englische Standard-Stopwörter (the, is, and, ...) raus,
# bevor das Vokabular gebaut wird.
vectorizer = TfidfVectorizer(stop_words='english')

# Reihenfolge der Klassen festhalten -> classes[i] gehört zu Zeile i der Matrix.
classes = list(class_doc.keys())

# fit:   lernt das Vokabular aus den 7 Dokumenten
# transform: wandelt die Texte in die TF-IDF-Matrix um (beides in einem Schritt)
# Ergebnis: sparse matrix mit Form (7 Klassen, ~58930 Wörter)
tfidf_matrix = vectorizer.fit_transform(class_doc.values())

# print(tfidf_matrix.shape)

# Array aller Wörter; die Reihenfolge entspricht exakt den Spalten der Matrix.
feature_names = vectorizer.get_feature_names_out()


# Für jede Klasse (= jede Zeile der Matrix) die Top-Wörter rausholen
for i in range(len(classes)):
    # tfidf_matrix[i]: die i-te Zeile (noch sparse)
    # .toarray():      sparse -> volles Array (alle ~58930 Werte, inkl. der vielen Nullen)
    # .flatten():      Form (1, 58930) -> flacher 1D-Vektor, mit dem argsort sauber arbeitet
    row = tfidf_matrix[i].toarray().flatten()

    # argsort gibt Indizes zurück,
    # die row klein -> groß sortieren würden. Größte Werte stehen also hinten.
    sorted_indices = row.argsort()

    # [-10:]   -> die letzten 10 Indizes = die 10 größten Werte
    # [::-1]   -> umdrehen, damit das stärkste Wort vorne steht
    top5 = sorted_indices[-10:][::-1]

    # Fancy Indexing: ein Array von Indizes greift alle 10 Wörter auf einmal
    # aus feature_names heraus (geht so nur bei numpy-Arrays, nicht bei Listen).
    top_words = feature_names[top5]

    # Klassenname + zugehörige Top-Wörter ausgeben
    print(classes[i], top_words)