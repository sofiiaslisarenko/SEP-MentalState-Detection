import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = os.path.join(os.getcwd(), "Datasets")
print(f"Path to datasets: {path}")
os.chdir(path)

file_get = [file for file in os.listdir(path)]
print(file_get)

# Explizit nach Dateiname laden statt nach Index (Reihenfolge ist nicht garantiert!)
df0 = pd.read_csv("Combined Data.csv")

print(df0.head())
print(df0.columns.tolist())
print(df0.nunique())

# Spalten bereinigen und umbenennen
df0_clean = df0[['statement', 'status']].rename(columns={'statement': 'text', 'status': 'label'})
df0_clean = df0_clean.dropna()

print(df0_clean['label'].value_counts())
print(f"Gesamt: {len(df0_clean)} Einträge")

X = df0_clean['text']
y = df0_clean['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Trainingsdaten: {len(X_train)} Einträge")
print(f"Testdaten: {len(X_test)} Einträge")

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Gespeichert: train.csv und test.csv")



