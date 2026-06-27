from datenhochladen import load_data_kaggle

# In zwei Funktionen aufgeteilt zum Bereinigen und zur Ausgabe

# TODO: "poll", "https" entfernen

def clean_data():
    df = load_data_kaggle()

    def remove_non_ascii(text):
        """Entfernt alle Nicht-ASCII-Zeichen (kaputte Mojibake, Emojis, Sonderzeichen).
        Begründung: Texte sind durchgehend englisch, Nicht-ASCII trägt kein Signal."""
        if not isinstance(text, str):
            return text
        return text.encode("ascii", errors="ignore").decode("ascii")

    df["statement"] = df["statement"].apply(remove_non_ascii)  # Entfernt alle Nicht-ASCII-Zeichen

    df = df.dropna(subset=['statement', 'status'])
    df = df.drop(columns=['Unnamed: 0'])

    # Entfernung von 13 Spalten mit NAME die leer sind
    df = df[df['statement'] != '#NAME?']

    return df

def print_clean_data():

    df = load_data_kaggle() # Der Datensatz vor Bereinigung
    df_clean = clean_data()

    mask = df_clean["statement"].str.contains("ð", na=False) # Überprüfen des Filters für nicht-ASCII Zeichen
    print("Mojibake-Treffer nach Filter:", mask.sum())

    # 1 Typen von Daten in jeder Spalte
    print("Datentypen:")
    print(df_clean.info())
    # 2 Gibt es leere Spalte
    print("Leere Columns:")
    print(df_clean.isnull().sum())
    # 3 Gibt es Reihen, die wiederholt wurden
    print("Anzahl von Duplikaten:", df_clean.duplicated().sum())
    # 4 Unique values werden statt 'status' die echte Name der Spalte
    print("Unique Classes:", df_clean['status'].unique())
    print(df_clean['status'].value_counts())
    # 5 Die leere Spalten werden gelöscht
    print("Groesse bevor Cleaning:", df.shape)
    print("Groesse nach Cleaning:", df_clean.shape)


if __name__ == "__main__":
    df = clean_data()
    print(df.head())