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

df0 = pd.read_csv(file_get[0])
#df1 = pd.read_csv(file_get[1])

print(df0.head())       #df0 enthält Combined Data.csv
#print(df1.head())       #df1 enthält Suicide_detection.csv

# cleanup und merging der beiden Dataframes, falls kompatibel

# Unique Werte in den Dataframes überprüfen, bzw Welche Werte sich Doppeln
# Und ob das für die Klassifikation relevant sein könnte, oder ob es sich um Rauschen handelt, welches entfernt werden sollte
print(df0.nunique())
#print(df1.nunique())

# Klassifikation zum finden, oder testen von Features, welche relevant sein könnten