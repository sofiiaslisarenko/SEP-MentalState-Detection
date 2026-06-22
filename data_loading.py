import os
import pandas as pd

def load_data():
    """Lädt den Hauptdatensatz aus dem Datasets-Ordner und gibt ihn als DataFrame zurück.

        Sucht die CSV-Datei relativ zum aktuellen Arbeitsverzeichnis im Unterordner
        'Datasets' und liest sie mit pandas ein.

        Returns:
            pd.DataFrame: Der eingelesene Datensatz (df0) mit den Spalten
                'statement' und 'status'.
        """
    # Finde den Pfad zu dem Datasets Ordner:
    path = os.getcwd()
    data_path = os.path.join(path, "Datasets")
    # Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
    os.chdir(data_path)

    # Erstellen einer Liste mit den csv Dateien und erste Auswertungen
    file_get = [file for file in os.listdir(data_path)]

    df0 = pd.read_csv(file_get[0])
    os.chdir(path)
    return df0

# Dieser Block läuft nur, wenn die Datei DIREKT gestartet wird (z.B. zum Testen),
# nicht beim Import durch andere Module wie eda.py.
if __name__ == "__main__":
    df0 = load_data()
    # Kurze Kontrolle, ob die Daten korrekt geladen wurden:
    print(df0.head())       # erste Zeilen ansehen
    print(df0.nunique())    # Anzahl eindeutiger Werte pro Spalte
