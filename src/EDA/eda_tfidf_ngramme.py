from datenbereinigung import clean_data
from training_test import train_testdaten_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import contractions
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words_nltk = set(stopwords.words('english'))

# FUNKTIONEN

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Kleinschreibung als Normalisierung. Erwartet bereits gesplittete Trainingsdaten."""
    df = df.copy()
    df['statement'] = df['statement'].str.lower()
    return df

def build_ignore_words(df: pd.DataFrame, stop_words, n: int = 20, threshold: int = 5) -> set[str]:
    """Wörter, die in >= threshold Klassen unter den Top-n vorkommen -> zu unspezifisch."""
    words_per_class = top_ngrams(df, stop_words, n=n, ngram=1)
    zaehler = Counter()
    for woerter in words_per_class.values():
        zaehler.update(woerter)
    return {word for word, count in zaehler.items() if count >= threshold}

def build_class_docs(df: pd.DataFrame) -> dict[str, str]:
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

# TF-IDF-MATRIX ÜBER ALLE 7 KLASSEN (STOPPWÖRTER ENTFERNT)

def tfidf_matrix_eda(df: pd.DataFrame, stop_words: set | None = None, ngram: int = 1) -> tuple:
    """
    Berechnet eine TF-IDF-Matrix auf Klassenebene (optional mit n-Grammen).

    Fügt pro Klasse alle Statements zu einem Dokument zusammen (über
    build_class_docs) und berechnet daraus eine TF-IDF-Matrix. Dadurch
    lassen sich die für jede Klasse typischsten Wörter bzw. Phrasen bestimmen.

    Bei ngram == 1 werden Unigramme mit Stopwortfilterung berechnet.
    Bei ngram > 1 werden ausschließlich n-Gramme der entsprechenden Länge
    ohne Stopwortfilterung berechnet (stop_words=None), damit Phrasen wie
    'do not want' erhalten bleiben.
    """
    # Pro Klasse alle Statements zu einem großen String zusammenfügen
    # -> ein "Dokument" pro Klasse
    class_doc = build_class_docs(df)

    # TfidfVectorizer wandelt die Texte in eine TF-IDF-Zahlenmatrix um;
    # stop_words entfernt die übergebenen Stopwörter vor dem Aufbau des Vokabulars.
    if ngram == 1:
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    else:
        vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(ngram, ngram))
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

def top_ngrams(df: pd.DataFrame, stop_words: set | None = None, n:int = 10, ngram: int = 1) -> dict:
    """Liefert pro Klasse die n Wörter mit dem höchsten TF-IDF-Wert.

    Berechnet die TF-IDF-Matrix auf Klassenebene und ermittelt für jede
    Klasse die typischsten Begriffe (höchste TF-IDF-Werte).

    Returns:
        dict[str, list[str]]: {Klassenname: [Top-n-Wörter absteigend]}
    """
    tfidf_matrix, classes, feature_names = tfidf_matrix_eda(df, stop_words, ngram=ngram)

    result = {}
    for i in range(len(classes)):
        row = tfidf_matrix[i].toarray().flatten()
        top_n = row.argsort()[-n:][::-1]
        # .tolist() -> echte Python-Liste statt NumPy-Array (umgeht das += Problem)
        result[classes[i]] = feature_names[top_n].tolist()

    return result

def print_top_ngrams(df, stop_words, n=10, ngram: int = 1) -> None:
    """Gibt pro Klasse die Top-n Wörter formatiert aus."""
    words_per_class = top_ngrams(df, stop_words, n, ngram)
    for klasse, woerter in words_per_class.items():
        print(f"--------------------- Top-Wörter in der Klasse {klasse} ---------------------")
        print(woerter)

# BALKENDIAGRAMME

def top_n_paare(row: np.ndarray, feature_names: np.ndarray,
                ignore_words: set[str], n: int = 20) -> list[tuple[str, float]]:
    """Liefert die Top-n (Wort, TF-IDF-Wert)-Paare einer Matrix-Zeile,
    nach Wert absteigend, ohne die Wörter aus ignore_words."""
    # Wörter + Werte paaren, dabei ignore_words rauslassen
    pairs = [(word, val) for word, val in zip(feature_names, row)
             if word not in ignore_words]
    # nach Wert (x[1]) absteigend sortieren, Top-n abschneiden
    top = sorted(pairs, key=lambda x: x[1], reverse=True)[:n]
    return top


def vergleich_klassen(df: pd.DataFrame, stop_words: set[str], ignore_words: set[str],
                      n: int = 20) -> None:
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
        # plt.savefig(f"../output/tfidf_compare/tfidf_compare_{classes[i]}.png")
        plt.show()


def plot_ngramms(df: pd.DataFrame, stop_words: set[str], n: int = 20, ngram: int = 1) -> None:
    """Zeigt pro Klasse ein horizontales Balkendiagramm der Top-n n-Gramme
    nach TF-IDF-Wert."""
    tfidf_matrix, classes, feature_names = tfidf_matrix_eda(df, stop_words, ngram)

    for i in range(len(classes)):
        row = tfidf_matrix[i].toarray().flatten()

        # Top-n Paare (ohne Filter -> leeres set)
        top = top_n_paare(row, feature_names, set(), n)

        # Wörter/Phrasen und Werte trennen
        words, vals = zip(*top)

        # ein Diagramm pro Klasse
        plt.figure(figsize=(10, 8))
        sns.barplot(x=list(vals), y=list(words), color="lightseagreen")
        plt.xlabel("TF-IDF-Wert")
        plt.title(f"Top-{n} {ngram}-Gramme nach TF-IDF – Klasse: {classes[i]}",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        # plt.savefig(f"../output/ngramme/{ngram}gramms_{classes[i]}.png")
        plt.show()

# ERSTELLEN EINES WORDCLOUDS

def word_clouds_image(df: pd.DataFrame, stop_words: set[str], ignore_words: set[str]) -> None:
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
        # plt.savefig(f"../output/word_clouds/WordCloud_{classes[i]}.png")
        plt.show()

# Häufigkeitsanalyse einer Einzelklasse (Depression) – Demonstration

def haeufigkeitsanalyse_depression(df: pd.DataFrame, stop_words: set[str]) -> None:
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

def run_eda_oleksandra(df: pd.DataFrame, stop_words: set[str]) -> None:
    """Orchestriert die komplette EDA. Wird aus main aufgerufen."""
    df = normalize_text(df)
    ignore_words = build_ignore_words(df, stop_words)

    # Hier die gewünschten Analysen aktivieren:
    # print_top_ngrams(df, stop_words, ngram=1)
    # plot_ngramms(df, stop_words, ngram=1)
    vergleich_klassen(df, stop_words, ignore_words, n=20)
    word_clouds_image(df, stop_words, ignore_words)
    # haeufigkeitsanalyse_depression(df, stop_words)

# KONTROLL-AUSGABE

if __name__ == "__main__":
    df = clean_data()
    df_train, _ = train_testdaten_split(df)
    run_eda_oleksandra(df_train, stop_words_nltk)
