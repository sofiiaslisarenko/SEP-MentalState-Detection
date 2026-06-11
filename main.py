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

df0 = pd.read(file_get[0])
dfSuicideDet = pd.read(file_get[1])