from data_clean import clean_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import contractions
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words_nltk = set(stopwords.words('english'))

# Importieren des DataFrames
df0 = clean_data()

# Normalisieren des Textes
df0['statement'] = df0['statement'].str.lower()

# FUNKTIONEN

def build_class_docs(df):
    """Fügt pro Klasse alle Statements zu einem großen String zusammen. Kontraktionen expandieren
    Gibt ein Dictionary zurück: {Klassenname: Gesamttext}."""
    # Dictionary anlegen, das pro Klasse den zusammengefügten Gesamttext speichert
    # (Schlüssel = Klassenname, Wert = ein großer String mit allen Statements der Klasse)
    class_doc = {}
    for cl in df['status'].unique():
        # Alle Statements herausfiltern, die zur aktuellen Klasse gehören.
        text = df[df['status'] == cl]['statement']
        # Kontraktionen expandieren (don't -- > do not, I'll --> I will)
        text_expanded = text.apply(contractions.fix)
        # Die einzelnen Statements zu einem großen String zusammenfügen, durch Leerzeichen getrennt
        # Den Gesamttext unter dem Klassennamen im Dictionary ablegen.
        class_doc[cl] = text_expanded.str.cat(sep=' ')
    return class_doc

# GENERELLE TF-IDF-MATRIX (STOPPWÖRTER ENTFERNT)

def tfidf_matrix_eda(df,stop_words):
    """Berechnet eine TF-IDF-Matrix auf Klassenebene.

        Fügt pro Klasse alle Statements zu einem Dokument zusammen (über
        build_class_docs) und berechnet daraus eine TF-IDF-Matrix. Dadurch
        lassen sich die für jede Klasse typischsten Wörter bestimmen.

        Args:
            df (pd.DataFrame): Datensatz mit den Spalten 'statement' und 'status'.
            stop_words_nltk (set | list): Stopwörter, die vor dem Vektorisieren
                entfernt werden (z.B. NLTK-Stopwortliste).

        Returns:
            tuple:
                - tfidf_matrix (scipy.sparse): Matrix der Form (n_klassen, n_woerter).
                - classes (list): Klassennamen; classes[i] gehört zu Zeile i.
                - feature_names (np.ndarray): Wörter; Index entspricht Spalte der Matrix.
        """
    # Pro Klasse alle Statements zu einem großen String zusammenfügen
    # -> ein "Dokument" pro Klasse
    class_doc = build_class_docs(df)

    # TfidfVectorizer wandelt die Texte in eine TF-IDF-Zahlenmatrix um;
    # stop_words entfernt die übergebenen Stopwörter vor dem Aufbau des Vokabulars.
    vectorizer = TfidfVectorizer(stop_words=list(stop_words_nltk))

    # fit:       lernt das Vokabular aus den Klassen-Dokumenten
    # transform: wandelt die Texte in die TF-IDF-Matrix um (beides in einem Schritt)
    # Ergebnis:  sparse matrix der Form (n_klassen, n_woerter)
    tfidf_matrix = vectorizer.fit_transform(class_doc.values())

    # Reihenfolge der Klassen festhalten -> classes[i] gehört zu Zeile i der Matrix
    classes = list(class_doc.keys())

    # Array aller Wörter; die Reihenfolge entspricht exakt den Spalten der Matrix
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, classes, feature_names

# TOP-WÖRTER PRO KLASSE (TF-IDF)

def top_words(df, stop_words, n=10):
    """Liefert pro Klasse die n Wörter mit dem höchsten TF-IDF-Wert.

    Berechnet die TF-IDF-Matrix auf Klassenebene und ermittelt für jede
    Klasse die typischsten Begriffe (höchste TF-IDF-Werte).

    Returns:
        dict[str, list[str]]: {Klassenname: [Top-n-Wörter absteigend]}
    """
    tfidf_matrix, classes, feature_names = tfidf_matrix_eda(df, stop_words)

    result = {}
    for i in range(len(classes)):
        row = tfidf_matrix[i].toarray().flatten()
        top_n = row.argsort()[-n:][::-1]
        # .tolist() -> echte Python-Liste statt NumPy-Array (umgeht das += Problem)
        result[classes[i]] = feature_names[top_n].tolist()

    return result

def print_top_words(df, stop_words, n=10):
    """Gibt pro Klasse die Top-n Wörter formatiert aus."""
    words_per_class = top_words(df, stop_words, n)
    for klasse, woerter in words_per_class.items():
        print(f"--------------------- Top-Wörter in der Klasse {klasse} ---------------------")
        print(woerter)

# Ertstellen von Liste der zusätzlichen Wörten, die ignoriert werden müssen.

words_per_class = top_words(df0, stop_words_nltk, n=20)

# 1. Alle Top-Wörter aller Klassen.
alle = []
for woerter in words_per_class.values():
    alle.extend(woerter)        # extend, nicht +=, und nicht append

# 2. Zählen, in wie vielen Klassen jedes Wort vorkam
zaehler = Counter(alle)

# Ausgabe der häufigsten Wörter
# for wort, anzahl in zaehler.most_common():
#     print(anzahl, wort)

# 3. Schwelle anwenden -> die comprehension schreibst du
ignore_words = {word for word, count in zaehler.items() if count >= 5}


# BALKENDIAGRAMME

def top_n_paare(row, feature_names, ignore_words, n=20):
    """Liefert die Top-n (Wort, TF-IDF-Wert)-Paare einer Matrix-Zeile,
    nach Wert absteigend, ohne die Wörter aus ignore_words."""
    # Wörter + Werte paaren, dabei ignore_words rauslassen
    pairs = [(word, val) for word, val in zip(feature_names, row)
             if word not in ignore_words]
    # nach Wert (x[1]) absteigend sortieren, Top-n abschneiden
    top = sorted(pairs, key=lambda x: x[1], reverse=True)[:n]
    return top


def vergleich_klassen(df, stop_words, ignore_words, n=20):
    """Zeigt für alle Klassen zwei Balkendiagramme nebeneinander:
    links ungefiltert, rechts mit ignore_words gefiltert."""
    tfidf_matrix, classes, feature_names = tfidf_matrix_eda(df, stop_words)

    for i in range(len(classes)):

        row = tfidf_matrix[i].toarray().flatten()

        # zweimal Top-n: einmal ohne Filter (leeres set), einmal mit
        top_ohne = top_n_paare(row, feature_names, set(), n)
        top_mit  = top_n_paare(row, feature_names, ignore_words, n)

        # zwei Subplots nebeneinander
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
        fig.suptitle(f"Top-{n} Wörter nach TF-IDF – Klasse: {classes[i]}", fontsize=16, fontweight="bold")

        # linkes Diagramm: ungefiltert
        words_ohne, vals_ohne = zip(*top_ohne)
        sns.barplot(x=list(vals_ohne), y=list(words_ohne), ax=axes[0], color="mediumpurple")
        axes[0].set_title("ungefiltert")
        axes[0].set_xlabel("TF-IDF-Wert")

        # rechtes Diagramm: gefiltert
        words_mit, vals_mit = zip(*top_mit)
        sns.barplot(x=list(vals_mit), y=list(words_mit), ax=axes[1], color="coral")
        axes[1].set_title("gefiltert")
        axes[1].set_xlabel("TF-IDF-Wert")

        plt.tight_layout()
        plt.show()


# ERSTELLEN EINES WORDCLOUDS

def word_clouds_image(df, stop_words, ignore_words):
    """Erzeugt und zeigt pro Klasse eine Word Cloud mit 100 Wörtern auf Basis der TF-IDF-Werte.

    Berechnet die TF-IDF-Matrix auf Klassenebene und stellt für jede Klasse
    eine eigene Word Cloud dar, in der die Wortgröße dem TF-IDF-Wert entspricht
    (nicht der rohen Häufigkeit). Die Wolken werden nacheinander als einzelne
    Bilder angezeigt.

    Es wurde die häufigsten Wörter trotzdem entfernt, die von nltk nicht abgefangen wurden.
    """
    # TF-IDF-Matrix samt Klassen- und Wortzuordnung holen
    tfidf_matrix, classes, feature_names = tfidf_matrix_eda(df, stop_words)

    # Pro Klasse (= Zeile der Matrix) eine Word Cloud erzeugen
    for i in range(len(classes)):
        # Sparse Zeile -> volles Array (alle Wörter inkl. Nullen)
        row = tfidf_matrix[i].toarray().flatten()

        # Dict {wort: tfidf_wert} bauen; generate_from_frequencies nutzt diese Werte
        # als Wortgrößen. Nullen werden von WordCloud ignoriert.
        freq = {wort: wert for wort, wert in zip(feature_names, row)
        if wort not in ignore_words}

        # Word Cloud aus den TF-IDF-Werten erzeugen
        # (generate_from_frequencies statt generate -> nutzt TF-IDF statt roher Häufigkeit)
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate_from_frequencies(freq)

        # Bild anzeigen: Achsen aus (bei einem Bild sinnlos), Klassenname als Titel
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(classes[i])
        plt.show()

# Häufigkeitsanalyse einer Einzelklasse (Depression) – Demonstration

def haeufigkeitsanalyse_depression(df, stop_words):
    """Zeigt: rohe Worthäufigkeiten sind von Stopwörtern dominiert, daher Filterung.
    Diese Analyse ist exemplarisch; die klassenübergreifende Auswertung folgt via TF-IDF."""

    # Nur die Depression-Statements herausfiltern
    depression_text = df[df['status'] == 'Depression']['statement']
    # print(depression_text.head())
    # print(depression_text.describe())

    # Alle Statements zu einem großen String verbinden, durch Leerzeichen getrennt
    big_text = depression_text.str.cat(sep=' ')
    # print(big_text)

    # Den String in einzelne Wörter zerlegen (an Leerzeichen)
    words = big_text.split()

    # Häufigkeiten zählen und die Top 20 ausgeben
    counter1 = Counter(words)
    # print(zaehler1.most_common(20))

    # Häufigkeiten zählen und die Top 20 ausgeben ohne Stoppwörter aus NLTK
    words_wo_stopwords = [word for word in words if word not in stop_words_nltk]
    counter2 = Counter(words_wo_stopwords)
    print("---------------------Top 20 Inhaltswörter in der Depression Klasse---------------------")
    for word, count in counter2.most_common(20):  # tuple unpacking
        print(f"{count:>10}  {word}")

# KONTROLL-AUSGABE

if __name__ == "__main__":
    # Klassen-Dokumente: pro Klasse einen kurzen Ausschnitt zur Inhaltsprüfung
    # class_doc = build_class_docs(df0)
    # for cl, text in class_doc.items():
    #     print(f"--------------------- Texte in der Klasse {cl} ---------------------")
    #     print()
    #     print(text[:300])
    #     print()

    # Top-Wörter pro Klasse über TF-IDF
    #print_top_words(df0, stop_words_nltk)

    # Häufigkeitsanalyse der Klasse Depression (Demonstration)
    #haeufigkeitsanalyse_depression(df0, stop_words_nltk)

    # Word Clouds pro Klasse (auskommentiert – öffnet 7 Bildfenster)
    #word_clouds_image(df0, stop_words_nltk, ignore_words)

    vergleich_klassen(df0, stop_words_nltk, ignore_words, n=20)