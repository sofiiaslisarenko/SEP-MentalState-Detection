import os
from contextlib import contextmanager
from datenbereinigung import clean_data, print_clean_data
from training_test import train_testdaten_split
from EDA.eda_tfidf_ngramme import run_eda_oleksandra, stop_words_nltk
from EDA.eda_structure_graphs import expl_data
from klassifikation_test import run_best_configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src

@contextmanager
def stable_cwd():
    """Setzt das CWD auf src/EDA, weil der EDA-Code seine Ausgaben über den
    relativen Pfad ../../output speichert – von src/EDA aus zeigt das auf den
    output-Ordner auf Projektebene. Legt output bei Bedarf an und stellt das
    ursprüngliche CWD danach garantiert wieder her, auch bei einem Fehler."""
    alter_cwd = os.getcwd()
    os.makedirs(os.path.join(BASE_DIR, "..", "output"), exist_ok=True)
    os.chdir(os.path.join(BASE_DIR, "EDA"))
    try:
        yield
    finally:
        os.chdir(alter_cwd)

# Daten Laden
df = clean_data()

# Informationen über Dataframe nach der Bereinigung ausgeben
print_clean_data(df)

# Daten für die EDA vorbereiten
df_train, _ = train_testdaten_split(df)

print("\n\n\n------------------------- Explorative Datenanalyse -------------------------\n\n\n")

# EDA von Oleksandra
# Gewünschte Analysen müssen in eda_tfidf_ngramme.py aktiviert werden!
run_eda_oleksandra(df_train, stop_words_nltk)

# EDA von Sebastian
with stable_cwd():
    expl_data(df_train.copy())
    # Weitere Funktionen

# EDA von Markus
    # Muss analog ergänzt werden

print("\n\n\n------------------------- Klassifikation -------------------------\n\n\n")

# Die gewählte beste Konfiguration für Klassifikation:
# Modelle: Logistic Regression, Random Forest
# TF-IDF mit angepasster Stoppwortliste (Negationen behalten), inkl. Bigramme
# Parameter class_weight="balanced" für beide Modelle
# Der Rest der Konfigurationen in klassifikation_test.py
run_best_configuration()

