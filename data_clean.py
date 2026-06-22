from data_loading import load_data


def clean_data():

    df = load_data()

    def remove_non_ascii(text):
        """Entfernt alle Nicht-ASCII-Zeichen (kaputte Mojibake, Emojis, Sonderzeichen).
        Begründung: Texte sind durchgehend englisch, Nicht-ASCII trägt kein Signal."""
        if not isinstance(text, str):
            return text
        return text.encode("ascii", errors="ignore").decode("ascii")

    df["statement"] = df["statement"].apply(remove_non_ascii) # Entfernt alle Nicht-ASCII-Zeichen

    mask = df["statement"].str.contains("ð", na=False) # Überprüfen des Filters für nicht-ASCII Zeichen
    print("Mojibake-Treffer nach Filter:", mask.sum())

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

if __name__ == "__main__":
    df = clean_data()
    print(df.head())