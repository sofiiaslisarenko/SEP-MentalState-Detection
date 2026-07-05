# Klassifizierung psychischer Zustände aus Textnachrichten

Dieses Projekt untersucht mit Methoden des maschinellen Lernens, wie sich Textnachrichten automatisch nach psychischen Zuständen klassifizieren lassen (Depression, Angst, Stress, Bipolar, Persönlichkeitsstörung, Suizidalität oder Normal). Der Code ist in einzelne Skripte gegliedert, die jeweils einen Arbeitsschritt abbilden – von der Datenbereinigung über die explorative Analyse bis zum Training und Vergleich verschiedener Modelle. Grundlage sind gelabelte Datensätze von Plattformen wie Reddit und Twitter sowie Methoden der Sentiment- und Textanalyse.

## Team

Gruppe 5:

- Uliana Buhaienko
- Markus Bühling
- Sebastian Lindner
- Oleksandra Maiboroda
- Sofiia Slisarenko

## Projektübersicht

- **Fragestellungen:**
- _Klassifikation:_
Lässt sich anhand eines Textes (statement) der psychische Zustand einer Person klassifizieren?
- _Regression:_
Lässt sich der mit VADER berechnete Sentiment-Score eines Textes aus einfachen, selbst definierten Textmerkmalen vorhersagen? 
- _Clustering:_
Welche natürlichen Gruppen lassen sich in den Texten finden, ohne Nutzung der vorhandenen Labels?

- **Ansatz:** Textvorverarbeitung → Feature Engineering (TF-IDF sowie linguistische Features → Training verschiedener Modellklassen → Vergleich und Evaluation
- **Datensatz:** Der Datensatz Combined Data (https://www.kaggle.com/datasets/szegeelim/mental-health/data) ist wurde von uns auf Kaggle gefunden und enthält
gelabelten Text von Socialmediaplatformen. Wie dem Namen zu entnehmen ist, wurde dieser aus anderen Kaggle Datensätzen zusammengesetzt.
Dem Text ist jeweils eins von 7 Labels zugeornet ('status'): Anxiety, Stress, Bipolar, Depression, Normal, Personality disorder, Suicidal.
Die Label sind keine medizinischen Diagnosen, sondern reflektieren lediglich den Kontext indem die Textbeiträge gefunden wurden z.B. bestimmte subreddits.

## Setup

```bash
git clone [REPO-URL]
cd [PROJEKTORDNER]
pip install -r requirements.txt
```

## Aufbau des Projekts

Der Quellcode liegt im Ordner `src/`. Der Datensatz (zum Anschauen) befindet sich in `Datasets/`, alle erzeugten Visualisierungen und Ergebnisse werden im Ordner `output/` abgelegt (unterteilt nach Analyseart, z.B. `clustering/`, `word_clouds/`, `VADER-Analyse/`).

> **Hinweis:** `src/` muss als Sources Root in PyCharm eingestellt werden (Rechtsklick auf `src/` → *Mark Directory as* → *Sources Root*).

### Datenaufbereitung

- **`datenhochladen.py`** – Lädt die Rohdaten von Kaggle ein und gibt sie als DataFrame zurück.
- **`datenbereinigung.py`** – Cleaning: entfernt fehlende Werte, Duplikate und kaputte Zeichen und gibt Informationen über den bereinigten Datensatz aus.
- **`feature_builder.py`** – Erzeugt aus dem bereinigten Text zwei Arten von Features: (1) Makro-Metriken pro Nachricht, etwa die Häufigkeit bestimmter Pronomen-Gruppen (`self_pronouns_count`, `first_pl_pr_count` …) und Fragezeichen, die als sprachliche Marker für psychische Zustände dienen können, sowie (2) Token-Frequenzen für definierte Ziel-Wörter. Die Funktion `create_all_features(df)` fügt alle berechneten Merkmale zu einem gemeinsamen DataFrame zusammen, der als Eingabe für die Modelle dient.
- **`training_test.py`** – Teilt die Daten in Trainings- und Testdatensatz auf (80/20, `random_state=42`).
- **`beispiel_instanzen.py`** – Ermöglicht das Überprüfen einzelner Beispieldaten.

### Explorative Datenanalyse (`EDA/`)

- **`eda_tfidf_ngramme.py`** – Untersucht die Texte mit TF-IDF und n-Grammen, erstellt Word Clouds für jede Klasse und andere Visualisierungen.
- **`eda_stopwordsratio_bigramme_vader.py`** – Analysiert Stoppwort-Anteil, Bigramme und VADER-Sentiment.

### Modellierung und Evaluation

- **`klassifikation.py`** / **`klassifikation_test.py`** / **`evaluation.py`**/ **`klassifikation_mit_Modellvisualisierung.py`** – Training und Evaluation der Klassifikationsmodelle.
- **`regression.py`** / **`regression_mit_modellvisualisierung.py`** – Training und Evaluation der Regressionsmodelle (letzteres inklusive Modellvisualisierung).
- **`clustering.py`** / **`clustering_visualisierung1.py`** – Untersuchung der Daten mit Clustering, Training und Bewertung des Clustering-Modells