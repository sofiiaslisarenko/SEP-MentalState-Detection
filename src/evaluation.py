import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

from datenbereinigung import clean_data
from feature_builder import create_all_features

# DATEN LADEN
df0 = clean_data()
df0, all_target_words = create_all_features(df0)

macro_features = [
    'word_count',
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
word_frequency_features = [col for col in df0.columns if col.startswith('freq_')]
feature_cols = macro_features + word_frequency_features
gewichtungs_faktor = 2

# ===== OVERFITTING =====
print("===== OVERFITTING (10% Trainingsdaten) =====")

train_df, test_df = train_test_split(df0, test_size=0.2, random_state=42, stratify=df0['status'])
train_df_small = train_df.sample(frac=0.1, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=3, max_df=0.7)
X_train_small_tfidf = tfidf.fit_transform(train_df_small['statement'])
X_test_tfidf = tfidf.transform(test_df['statement'])

scaler = StandardScaler()
X_train_small_num = scaler.fit_transform(train_df_small[feature_cols]) * gewichtungs_faktor
X_test_num = scaler.transform(test_df[feature_cols]) * gewichtungs_faktor

X_train_small_comb = hstack([csr_matrix(X_train_small_num), X_train_small_tfidf])
X_test_comb = hstack([csr_matrix(X_test_num), X_test_tfidf])

model_overfit = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_overfit.fit(X_train_small_comb, train_df_small['status'])

print(f"Accuracy auf Trainingsdaten: {accuracy_score(train_df_small['status'], model_overfit.predict(X_train_small_comb)):.2f}")
print(f"Accuracy auf Testdaten:      {accuracy_score(test_df['status'], model_overfit.predict(X_test_comb)):.2f}")

# ===== UNDERFITTING =====
print("\n===== UNDERFITTING (sehr einfache Logistic Regression) =====")

tfidf2 = TfidfVectorizer(max_features=5000, stop_words='english', min_df=3, max_df=0.7)
X_train_tfidf = tfidf2.fit_transform(train_df['statement'])
X_test_tfidf2 = tfidf2.transform(test_df['statement'])

scaler2 = StandardScaler()
X_train_num = scaler2.fit_transform(train_df[feature_cols]) * gewichtungs_faktor
X_test_num2 = scaler2.transform(test_df[feature_cols]) * gewichtungs_faktor

X_train_comb = hstack([csr_matrix(X_train_num), X_train_tfidf])
X_test_comb2 = hstack([csr_matrix(X_test_num2), X_test_tfidf2])

model_underfit = LogisticRegression(max_iter=5, C=0.001, solver='lbfgs')
model_underfit.fit(X_train_comb, train_df['status'])
print(f"Accuracy auf Testdaten: {accuracy_score(test_df['status'], model_underfit.predict(X_test_comb2)):.2f}")

# EXTREME SPLITS
def test_extreme_split(df0, feature_cols, test_size, n=10):
    accuracies = []
    for i in range(n):
        tr, te = train_test_split(df0, test_size=test_size, random_state=i, stratify=df0['status'])

        tfidf_ex = TfidfVectorizer(max_features=5000, stop_words='english', min_df=3, max_df=0.7)
        X_tr_tfidf = tfidf_ex.fit_transform(tr['statement'])
        X_te_tfidf = tfidf_ex.transform(te['statement'])

        scaler_ex = StandardScaler()
        X_tr_num = scaler_ex.fit_transform(tr[feature_cols]) * gewichtungs_faktor
        X_te_num = scaler_ex.transform(te[feature_cols]) * gewichtungs_faktor

        X_tr_comb = hstack([csr_matrix(X_tr_num), X_tr_tfidf])
        X_te_comb = hstack([csr_matrix(X_te_num), X_te_tfidf])

        model = LogisticRegression(max_iter=2000, C=10, class_weight='balanced', solver='lbfgs')
        model.fit(X_tr_comb, tr['status'])
        accuracies.append(accuracy_score(te['status'], model.predict(X_te_comb)))

    print(f"\nTest Size {test_size} ({n} Splits):")
    print(f"  Durchschnitt: {np.mean(accuracies):.3f}")
    print(f"  Varianz:      {np.var(accuracies):.6f}")

print("\n===== EXTREME TRAIN-TEST SPLITS =====")
test_extreme_split(df0, feature_cols, test_size=0.1)
test_extreme_split(df0, feature_cols, test_size=0.9)