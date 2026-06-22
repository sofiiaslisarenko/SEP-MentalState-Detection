import os
import pandas as pd
import kagglehub

def load_data():
    """Lädt den Hauptdatensatz aus dem Datasets-Ordner und gibt ihn als DataFrame zurück.

        Sucht die CSV-Datei relativ zum aktuellen Arbeitsverzeichnis im Unterordner
        'Datasets' und liest sie mit pandas ein.

        Returns:
            pd.DataFrame: Der eingelesene Datensatz (df0) mit den Spalten
                'statement' und 'status'.
        """
    # Finde den Pfad zu dem Datasets Ordner:
    path = os.path.join(os.getcwd(), "Datasets")

    # Ändere das Arbeitsverzeichnis in den gefundenen Pfad:
    os.chdir(path)

    # Erstellen einer Liste mit den csv Dateien und erste Auswertungen
    file_get = [file for file in os.listdir(path)]

    df = pd.read_csv(file_get[0], encoding='utf-8')

    return df


def load_data_kaggle():
    """Lädt den Hauptdatensatz direkt von Kaggle und gibt ihn als DataFrame zurück.

    Lädt das Dataset 'szegeelim/mental-health' über kagglehub herunter
    (wird lokal gecached) und liest die enthaltene CSV-Datei ein.

    Returns:
        pd.DataFrame: Der eingelesene Datensatz mit den Spalten
            'statement' und 'status'.
    """
    # Dataset herunterladen; kagglehub gibt den Pfad zum Cache-Ordner zurück
    path = kagglehub.dataset_download("szegeelim/mental-health")

    csv_path = os.path.join(path, "Combined Data.csv")  # exakter Dateiname

    df = pd.read_csv(csv_path, encoding="utf-8")

    return df


# Dieser Block läuft nur, wenn die Datei DIREKT gestartet wird (z.B. zum Testen),
# nicht beim Import durch andere Module wie eda.py.
if __name__ == "__main__":
    df0 = load_data()
    df1 = load_data_kaggle()
    # Kurze Kontrolle, ob die Daten korrekt geladen wurden:
    # print(df0.head())       # erste Zeilen ansehen
    # print(df0.nunique())    # Anzahl eindeutiger Werte pro Spalte
    print(df1.head())

    kaputt = df0["statement"].iloc[348]   # die Zeile mit dem internship-Emoji
    print(repr(kaputt)) # Kaputte Zeichen - 'â', 'ðŸ'
    mask = df0["statement"].str.contains("ð", na=False)
    print("Mojibake-Treffer:", mask.sum())