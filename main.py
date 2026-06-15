import pandas as pd
import os
from sklearn.model_selection import train_test_split

#Finde den Pfad zu dem Datasets Ordner:
path = os.path.join(os.getcwd(), "Datasets")
print(f"Path to datasets: {path}")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Erstellen einer Liste mit den csv Dateien und erste Auswertungen
file_get = [file for file in os.listdir(path)]
print(file_get)

df0 = pd.read_csv(file_get[0])
df1 = pd.read_csv(file_get[1])

print(df0.head())       #df0 enthält Combined Data.csv
print(df1.head())       #df1 enthält Suicide_detection.csv

# cleanup und merging der beiden Dataframes, falls kompatibel

# Unique Werte in den Dataframes überprüfen, bzw Welche Werte sich Doppeln
# Und ob das für die Klassifikation relevant sein könnte, oder ob es sich um Rauschen handelt, welches entfernt werden sollte
print(df0.nunique())
print(df1.nunique())

# Klassifikation zum finden, oder testen von Features, welche relevant sein könnten


# Nur die relevanten Spalten aus df0 auswählen und umbenennen,
# damit beide Dataframes die gleichen Spaltennamen haben
df0_clean = df0[['text', 'class']].rename(columns={'text': 'text', 'class': 'label'})

# Gleiches für df1 - 'statement' wird zu 'text', 'status' wird zu 'label'
df1_clean = df1[['statement', 'status']].rename(columns={'statement': 'text', 'status': 'label'})

# Beide Dataframes zu einem großen Dataframe zusammenfügen
# ignore_index=True setzt die Zeilennummern neu (0, 1, 2, ...)
df_combined = pd.concat([df0_clean, df1_clean], ignore_index=True)

# Zeilen mit fehlenden Werten (leere Texte oder fehlende Labels) entfernen
df_combined = df_combined.dropna()

# Übersicht: wie viele Einträge gibt es pro Klasse?
print(df_combined['label'].value_counts())
print(f"Gesamt: {len(df_combined)} Einträge")

#----------------------------------
# Train/Test Split (80% train, 20% test)
X = df_combined['text'] # X enthält die Eingabedaten (Texte), die das Modell lesen wird
y = df_combined['label'] # y enthält die richtigen Antworten (Labels wie 'suicide', 'Anxiety', ...)

# Aufteilen in Trainings- (80%) und Testdatensatz (20%)
# random_state=42 sorgt dafür, dass die Aufteilung immer gleich ist
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} Einträge")
print(f"Test: {len(X_test)} Einträge")

# Trainings- und Testdaten als CSV-Dateien speichern
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Gespeichert: train.csv und test.csv")