import pandas as pd
from datenbereinigung import clean_data
from feature_builder import create_all_features
from training_test import train_testdaten_split, train_testdaten_split_no_stratify
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')  # Laedt das VADER-Lexikon herunter (nur beim ersten Mal noetig)

# VADER-Werkzeug einmal erzeugen und dann wiederverwenden
analyzer = SentimentIntensityAnalyzer()

# Daten laden und bereinigen
df0 = clean_data()

# Handgemachte Features berechnen (Pronomen-, Wort- und Satzzeichen-Zaehler)
df0, all_target_words = create_all_features(df0)

# Zielvariable erzeugen: VADER-Sentiment-Score (-1 bis 1) pro Statement
df0['sentiment'] = df0['statement'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Aufteilen in Train/Test ohne stratify (Zielgroesse ist kontinuierlich, nicht klassenbasiert)
train_df, test_df = train_testdaten_split_no_stratify(df0)

# Makro-Features auswaehlen (auskommentierte werden bewusst nicht genutzt)
macro_features = [
    'word_count',
    #'pronoun_dominance',
    #'absolutist_ratio',
    #'absolute_uncertain_ratio',
    #'time_focus_score',
    #'future_count',
    #'past_count',
    'self_pronouns_count',
    'first_pl_pr_count',
    'second_pronouns_count',
    'third_pr_count',
    'other_pl_pr_count',
    'self_pr_other_count',
    'question_marks_count',
    'ellipses_count',
    'exclamation_marks_count',
    'sleep_words'
]

# Alle Wort-Haeufigkeits-Spalten automatisch einsammeln
word_frequency_features = [col for col in df0.columns if col.startswith('freq_')]

# Vollstaendige Feature-Liste fuer das Modell
feature_cols = macro_features + word_frequency_features

print(f"Das Modell hat jetzt {len(feature_cols)} Features")
print('sentiment' in feature_cols)  # Pruefung: Ziel darf nicht in den Features sein (muss False sein)

# X = Eingangs-Features, y = Zielwert (Sentiment), getrennt fuer Train und Test
X_train = train_df[feature_cols]
y_train = train_df['sentiment']
X_test  = test_df[feature_cols]
y_test  = test_df['sentiment']

# Features skalieren: nur auf Train fitten, auf beide anwenden (verhindert Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Modell 1: Ridge (lineare Regression mit Regularisierung)
model_ridge = Ridge()
model_ridge.fit(X_train_scaled, y_train)
y_pred_ridge = model_ridge.predict(X_test_scaled)

# Modell 2: Random Forest (nicht-lineares Modell aus vielen Entscheidungsbaeumen)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)

# Auswertung: R2 (erklaerte Varianz) und MAE (mittlerer Fehler) pro Modell
print("===== Ridge =====")
print(f"R²:  {r2_score(y_test, y_pred_ridge):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.3f}")

print("===== Random Forest =====")
print(f"R²:  {r2_score(y_test, y_pred_rf):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.3f}")
