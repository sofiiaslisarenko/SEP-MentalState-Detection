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
for sets in file_get:
    print("=" * 80)
    print("FILE:", os.path.basename(sets))

    df = pd.read_csv(sets)

    # Show columns first
    print("\nColumns:", df.columns.tolist())

    # Try to show value counts for ALL text-like columns
    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        print(f"\n🔹 Value counts for column: '{col}'")
        print(df[col].value_counts())