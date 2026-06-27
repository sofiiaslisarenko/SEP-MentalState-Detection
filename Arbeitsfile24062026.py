import pandas as pd
import os

#Finde den Pfad zu dem Datasets Ordner:
path = os.path.join(os.getcwd(), "Datasets")
print(f"Path to datasets: {path}")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Erstellen einer Liste mit den csv Dateien und erste Auswertungen
file_get = [file for file in os.listdir(path)]
print(file_get)
"""for sets in file_get:
    print("=" * 80)
    print("FILE:", os.path.basename(sets))

    df = pd.read_csv(sets)

    # Show columns first
    print("\nColumns:", df.columns.tolist())

    # Try to show value counts for ALL text-like columns
    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        print(f"\n🔹 Value counts for column: '{col}'")
        print(df[col].value_counts())"""
# ----------------------------------- Mein Code ---------------------------------
df0 = pd.read_csv(file_get[0])
df1 = pd.read_csv(file_get[1])

import nltk
from nltk.corpus import stopwords
import string # wichtig für spätere Entfernung der Satzzeichen

nltk.download('stopwords', quiet=True) # englische Stopwörter sind wichtig für das weitere Vorgehen
# quiet=True -> es werden keine download infos angezeigt
# Optimierung für beide Textprocessingmethoden, da beide noch zu langsam sind
stw_eng = set(stopwords.words('english'))
satzzeichentabelle = str.maketrans('', '', string.punctuation)


def text_process(mess):
    """
    Bereinigt einen Textstring durch folgende Schritte:
    1. Entfernt alle Satzzeichen
    2. Entfernt alle Stoppwörter
    3. Gibt eine Liste des gereinigten Textes zurück
    """
    # 1.
    text_ohne_sz = mess.translate(satzzeichentabelle)
    clea = []
    for wort in text_ohne_sz.split():
        wort_lower = wort.lower()
        # 2.
        if wort_lower not in stw_eng:
            clea.append(wort_lower)
            #3.
    return clea



def text_process_with_stw(mess):
    """
    Bereinigt einen Textstring durch folgende Schritte:
    1. Entfernt alle Satzzeichen
    2. Entfernt NICHT die Stoppwörter
    3. Gibt eine Liste des gereinigten Textes zurück
    """
    # 1.
    text_ohne_sz = mess.translate(satzzeichentabelle)
    clea = []
    for wort in text_ohne_sz.split():
        clea.append(wort.lower())
            # 3.
    return clea

print("\n =========== Meine Analyse ===========")
# ---------   df1
print("============================   df0   ============================")
print("head of df0\n",df0.head(3))
print()
print("tail of df0\n",df0.tail(3))

print()

form = df0.shape # shape liefert einen Tupel zurück (Spalten, Zeilen)
print(f"Form: {form}\nDer Datensatz df0 hat {form[0]} Zeilen und {form[1]} Spalten.\n")
df0.info() # gibt Auskunft über column Namen, datatype, Zahl der Einträge
print("\nFehlende Werte in df0:\n", df0.isnull().sum())

print()
print("---------------------- df0 mit neuen Spalten --------------------")
print()
df0['statement']= df0['statement'].fillna("").astype(str)
df0['tokenized_text'] = df0['statement'].apply(text_process)
df0['tokenized_text_with_stw'] = df0['statement'].apply(text_process_with_stw)
anz_ohne_stw= df0['tokenized_text'].str.len()
anz_mit_stw= df0['tokenized_text_with_stw'].str.len()
df0['stopwords_per_text_ratio'] = (anz_mit_stw - anz_ohne_stw) / anz_mit_stw.replace({0:1})


print("head of df0\n",df0.head(3))
print()
print("tail of df0\n",df0.tail(3))


form = df0.shape # shape liefert einen Tupel zurück (Spalten, Zeilen)
print(f"Form: {form}\nDer Datensatz df0 hat {form[0]} Zeilen und {form[1]} Spalten.\n")
df0.info() # gibt Auskunft über column Namen, datatype, Zahl der Einträge
print("\nFehlende Werte in df0:\n", df0.isnull().sum())
print()
auswertung0 = df0.groupby('status')['stopwords_per_text_ratio'].describe()
print("---------------------    Auswertung zu df0   -------------------")
print(auswertung0)

#---------  df1
print("============================   df1   ============================")

print("head of df1\n",df1.head(3))
print()
print("tail of df1\n",df1.tail(3))

print()
print("df1")
form = df1.shape # shape liefert einen Tupel zurück (Spalten, Zeilen)
print(f"Form: {form}\nDer Datensatz df1 hat {form[0]} Zeilen und {form[1]} Spalten.\n")
df1.info() # gibt Auskunft über column Namen, datatype, Zahl der Einträge
print("\nFehlende Werte in df1:\n", df1.isnull().sum())
print()
print("---------------------- df1 mit neuen Spalten --------------------")


df1['text']= df1['text'].fillna("").astype(str)
df1['tokenized_text'] = df1['text'].apply(text_process)
df1['tokenized_text_with_stw'] = df1['text'].apply(text_process_with_stw)
anz_ohne_stw1= df1['tokenized_text'].str.len()
anz_mit_stw1= df1['tokenized_text_with_stw'].str.len()
df1['stopwords_per_text_ratio'] = (anz_mit_stw1 - anz_ohne_stw1) / anz_mit_stw1.replace({0:1})

print()
form = df1.shape # shape liefert einen Tupel zurück (Spalten, Zeilen)
print(f"Form: {form}\nDer Datensatz df1 hat {form[0]} Zeilen und {form[1]} Spalten.\n")
df1.info() # gibt Auskunft über column Namen, datatype, Zahl der Einträge
print("Fehlende Werte in df1:\n", df1.isnull().sum())
print()
auswertung1 = df1.groupby('class')['stopwords_per_text_ratio'].describe()
print("---------------------    Auswertung zu df1 -------------------")
print(auswertung1)



# ================================= Unigramme und Bigramme =================================

from sklearn.feature_extraction.text import CountVectorizer

def zeige_top_ngramme(dataframe, text_spalte, ziel_spalte, n=1, top_k=10):
    """Zeigt die häufigsten N-Gramme pro Klasse an.
    n=1 steht für Unigramme, n=2 für Bigramme."""

    print(f"\n================ Top {top_k} {'Unigramme' if n == 1 else 'Bigramme'} pro Klasse ================")

    # Einzigartige Klassen herausholen
    klassen = dataframe[ziel_spalte].unique()

    for klasse in klassen:
        # 1. Daten für die aktuelle Klasse filtern
        klassen_texte = dataframe[dataframe[ziel_spalte] == klasse][text_spalte]

        # Falls keine Texte vorhanden sind, überspringen
        if len(klassen_texte) == 0:
            continue

        # 2. CountVectorizer initialisieren (wir nutzen die bereits bereinigten tokenisierten Listen)
        # Da die Spalte 'tokenized_text' Listen enthält, fügen wir die Wörter temporär per join zusammen
        texte_als_string = klassen_texte.apply(lambda x: " ".join(x))

        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=top_k * 2)
        try:
            bag_of_words = vectorizer.fit_transform(texte_als_string)

            # Häufigkeiten aufsummieren
            wort_summen = bag_of_words.sum(axis=0)
            woerter_haeufigkeit = [(wort, wort_summen[0, idx]) for wort, idx in vectorizer.vocabulary_.items()]

            # Sortieren nach Häufigkeit
            woerter_haeufigkeit = sorted(woerter_haeufigkeit, key=lambda x: x[1], reverse=True)

            # Ausgabe formatieren
            print(f"\nKategorie:  <{klasse}>  (Gesamt: {len(klassen_texte)} Posts)")
            for wort, count in woerter_haeufigkeit[:top_k]:
                print(f"  -> {wort:<20} : {count} Mal")

        except ValueError:
            # Falls eine Klasse nur aus leeren Dokumenten besteht
            print(f"\nKategorie:  <{klasse}>  - Keine ausreichenden Wörter für die Analyse.")

# --- Aufruf für df0 ---
# Wichtig: Wir nehmen die bereits gesäuberte Spalte 'tokenized_text'
print()
"""
print()
print("Unigramme df0")
zeige_top_ngramme(df0, text_spalte='tokenized_text', ziel_spalte='status', n=1, top_k=10)  # Unigramme
print()
print()
print("Bigramme df0 - Stoppwörter nicht enthalten")
zeige_top_ngramme(df0, text_spalte='tokenized_text', ziel_spalte='status', n=2, top_k=10)  # Bigramme
print()
print()
print("Bigramme df0 - Stoppwörter inklusive")
zeige_top_ngramme(df0, text_spalte='tokenized_text_with_stw', ziel_spalte='status', n=2, top_k=10)  # Bigramme Stoppwörter inklusive
print()
print()
print("Unigramme df1")
zeige_top_ngramme(df1, text_spalte='tokenized_text', ziel_spalte='class', n=1, top_k=10)  # Unigramme
print()
print()
print("Bigramme df1 - Stoppwörter nicht enthalten")
zeige_top_ngramme(df1, text_spalte='tokenized_text', ziel_spalte='class', n=2, top_k=10)  # Bigramme
print()
print()
print("Bigramme df1 - Stoppwörter inklusive")
zeige_top_ngramme(df1, text_spalte='tokenized_text_with_stw', ziel_spalte='class', n=2, top_k=10)  # Bigramme Stoppwörter inklusive
"""
# =============  DARTH VADER  ============
#VADER versteht nur Englisch (in unserem Fall kein Problem)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Vader-Lexikon-Katalog herunterladen
nltk.download('vader_lexicon', quiet=True)
# VADER Analyser initialisieren
sia = SentimentIntensityAnalyzer()
# Anwendung auf unverarbeiteten Text, hier brauchen wir erstmal nur den kombinierten Gesamtwert (compound)
# neben compound gäbe es noch pos, neu, neg (möglicherweise interessant für unsere features)

#  -------------------------- df0 ---------------------------
#df0['vader_compound']= df0['statement'].apply(lambda text: sia.polarity_scores(str(text))['compound'])

#print("\n Durchschnittliches Sentiment pro Kategorie in df0:")
#print(df0.groupby('status')['vader_compound'].mean())

#  -------------------------- df1 ---------------------------
df1['vader_compound']= df1['text'].apply(lambda text: sia.polarity_scores(str(text))['compound'])

print("\n Durchschnittliches Sentiment pro Kategorie in df1:")
print(df1.groupby('class')['vader_compound'].mean())

# Visualisierung

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Berechnen der VADER-Mittelwerte pro Kategorie (wie gehabt)
# Wir speichern das Ergebnis in einer neuen Variable 'vader_means'
vader_means = df1.groupby('class')['vader_compound'].mean().sort_values()

# 2. Diagramm-Stil definieren (Seaborn sorgt für ein schönes Design)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))  # Breite und Höhe der Grafik in Zoll

# 3. Balkendiagramm erstellen
# Wir nutzen eine Farbpalette, die von rot (negativ) nach blau (positiv) verläuft
colors = sns.color_palette("RdYlBu", len(vader_means))
barplot = sns.barplot(x=vader_means.index, y=vader_means.values, palette=colors)

# 4. Titel und Beschriftungen hinzufügen
plt.title('Durchschnittliches VADER-Sentiment pro psychologischer Kategorie', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Kategorie (class)', fontsize=12, labelpad=10)
plt.ylabel('Durchschnittlicher Sentiment-Score (Compound)', fontsize=12, labelpad=10)

# 5. Y-Achse begrenzen, da VADER-Werte immer zwischen -1 und +1 liegen
plt.ylim(-1.0, 1.0)

# Eine feine Linie bei 0 einfügen zur besseren Orientierung zwischen positiv/negativ
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

# 6. Die exakten Zahlenwerte über/unter die Balken schreiben
for bar in barplot.patches:
    y_val = bar.get_height()
    # Platzierung des Textes leicht anpassen, je nachdem ob der Balken positiv oder negativ ist
    va_dir = 'bottom' if y_val >= 0 else 'top'
    offset = 0.03 if y_val >= 0 else -0.05

    barplot.text(
        bar.get_x() + bar.get_width() / 2,  # X-Position (Mitte des Balkens)
        y_val + offset,  # Y-Position
        f'{y_val:.3f}',  # Der Text (auf 3 Nachkommastellen gerundet)
        ha='center',  # Horizontale Ausrichtung
        va=va_dir,  # Vertikale Ausrichtung
        fontsize=10,
        fontweight='bold'
    )

# 7. Layout optimieren und Diagramm anzeigen
plt.tight_layout()
plt.show()