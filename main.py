def clean_data(df):
    #Herunterladung von Dataset, damit man das bearbeiten und prüfen kann

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("szegeelim/mental-health")

    print("Path to dataset files:", path)
    import kagglehub
    import os
    import pandas as pd
    path = kagglehub.dataset_download("szegeelim/mental-health")
    # Weg zu dem File genau
    file_path = os.path.join(path, "Combined Data.csv")
    # Wird als Tabelle gelesen
    df = pd.read_csv(file_path)
    # 1 Typen von Daten in jeder Spalte
    print("Datentypen:")
    print(df.info())
    # 2 Gibt es leere Spalte
    print("Leere Columns:")
    print(df.isnull().sum())
    # 3 Gibt es Reihen, die wiederholt wurden
    print("Anzahl von Duplikaten:", df.duplicated().sum())
    # 4 Unique values werden statt 'status' die echte Name der Spalte
    print("Unique Classes:", df['status'].unique())
    print(df['status'].value_counts())
    # 5 Die leere Spalten werden gelöscht
    print("Groesse bevor Cleaning:", df.shape)
    df = df.dropna(subset=['statement', 'status'])
    print("Groesse nach Cleaning:", df.shape)

    # Entfernung von 13 Spalten mit NAME die leer sind
    df = df[df['statement'] != '#NAME?']
    print("Groesse mit NAME:", df.shape)
    print(df[df['statement'].str.contains('NAME', na=False)]['statement'].tolist())
    return df