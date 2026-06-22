from data_clean import clean_data
import matplotlib.pyplot as plt
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

# ============================== FUNKTIONEN ==============================

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

def tfidf_matrix_general(df,stop_words):
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

# ERSTELLEN EINES WORDCLOUDS

def word_clouds_image(df, stop_words):
    """Erzeugt und zeigt pro Klasse eine Word Cloud auf Basis der TF-IDF-Werte.

    Berechnet die TF-IDF-Matrix auf Klassenebene und stellt für jede Klasse
    eine eigene Word Cloud dar, in der die Wortgröße dem TF-IDF-Wert entspricht
    (nicht der rohen Häufigkeit). Die Wolken werden nacheinander als einzelne
    Bilder angezeigt.
    """
    # TF-IDF-Matrix samt Klassen- und Wortzuordnung holen
    tfidf_matrix, classes, feature_names = tfidf_matrix_general(df, stop_words)

    # Pro Klasse (= Zeile der Matrix) eine Word Cloud erzeugen
    for i in range(len(classes)):
        # Sparse Zeile -> volles Array (alle Wörter inkl. Nullen)
        row = tfidf_matrix[i].toarray().flatten()

        # Dict {wort: tfidf_wert} bauen; generate_from_frequencies nutzt diese Werte
        # als Wortgrößen. Nullen werden von WordCloud ignoriert.
        freq = dict(zip(feature_names, row))

        # Word Cloud aus den TF-IDF-Werten erzeugen
        # (generate_from_frequencies statt generate -> nutzt TF-IDF statt roher Häufigkeit)
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)

        # Bild anzeigen: Achsen aus (bei einem Bild sinnlos), Klassenname als Titel
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(classes[i])
        plt.show()

# === Häufigkeitsanalyse einer Einzelklasse (Depression) – Demonstration ===

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
    # big_text = big_text.lower() # TODO: mit Team klären – lower() zentral in cleaning oder lokal?

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

# TOP-WÖRTER PRO KLASSE (TF-IDF)

def print_top_words(df, stop_words, n=20):
    """Gibt pro Klasse die n Wörter mit dem höchsten TF-IDF-Wert aus.

    Berechnet die TF-IDF-Matrix auf Klassenebene und ermittelt für jede
    Klasse die Wörter mit den höchsten Werten – also die für die Klasse
    typischsten Begriffe.

    Args:
        df (pd.DataFrame): Datensatz mit den Spalten 'statement' und 'status'.
        stop_words (set | list): Stopwörter, die vor dem Vektorisieren entfernt werden.
        n (int): Anzahl der auszugebenden Top-Wörter pro Klasse (Standard: 20).
    """
    # TF-IDF-Matrix samt Klassen- und Wortzuordnung holen
    tfidf_matrix, classes, feature_names = tfidf_matrix_general(df, stop_words)

    # Für jede Klasse (= jede Zeile der Matrix) die Top-Wörter rausholen
    for i in range(len(classes)):
        # sparse Zeile -> volles Array, damit argsort damit arbeiten kann
        row = tfidf_matrix[i].toarray().flatten()

        # Indizes klein -> groß sortiert; die n größten hinten, dann umdrehen
        top_n = row.argsort()[-n:][::-1]

        # Wörter zu diesen Indizes holen (Fancy Indexing auf dem numpy-Array)
        top_words = feature_names[top_n]

        print(f"--------------------- Top-Wörter in der Klasse {classes[i]} ---------------------")
        print(top_words)
# ============================== KONTROLL-AUSGABE ==============================

if __name__ == "__main__":
    # Klassen-Dokumente: pro Klasse einen kurzen Ausschnitt zur Inhaltsprüfung
    class_doc = build_class_docs(df0)
    for cl, text in class_doc.items():
        print(f"--------------------- Texte in der Klasse {cl} ---------------------")
        print()
        print(text[:300])
        print()

    # Top-Wörter pro Klasse über TF-IDF
    print_top_words(df0, stop_words_nltk)

    # Häufigkeitsanalyse der Klasse Depression (Demonstration)
    haeufigkeitsanalyse_depression(df0, stop_words_nltk)

    # Word Clouds pro Klasse (auskommentiert – öffnet 7 Bildfenster)
    # word_clouds_image(df0, stop_words_nltk)