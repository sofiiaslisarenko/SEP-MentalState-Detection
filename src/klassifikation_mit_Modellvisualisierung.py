import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import pandas as pd
from datenbereinigung import clean_data
from feature_builder import create_all_features
from training_test import train_testdaten_split
import numpy as np

df0 = clean_data()
df0, all_target_words = create_all_features(df0)
train_df, test_df = train_testdaten_split(df0)

macro_features = [
    'word_count',
    'all_pronouns',
    'absolutist_count',
    'uncertain_count',
    'future_count',
    'past_count',
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


X_train = train_df[feature_cols]
y_train = train_df['status']
X_test = test_df[feature_cols]
y_test = test_df['status']

# TF-IDF AUS DEM STATEMENT-TEXT

tfidf = TfidfVectorizer(max_features=5000, stop_words='english',min_df = 3, max_df=0.7)
X_train_tfidf = tfidf.fit_transform(train_df['statement'])
X_test_tfidf = tfidf.transform(test_df['statement'])


# SKALIERUNG DER NUMERISCHEN FEATURES

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train)
X_test_num = scaler.transform(X_test)
gewichtungs_faktor = 2

X_train_num *= gewichtungs_faktor
X_test_num *= gewichtungs_faktor

# KOMBINIEREN: NUMERISCHE FEATURES + TF-IDF

X_train_combined = hstack([csr_matrix(X_train_num), X_train_tfidf])
X_test_combined = hstack([csr_matrix(X_test_num), X_test_tfidf])


# MODELL 1: LOGISTIC REGRESSION
print("\n\n\n Testen nur mit eingebauten Stoppwörtern: ")
model_lr = LogisticRegression(max_iter=2000, C = 10, class_weight="balanced", solver = 'lbfgs')
model_lr.fit(X_train_combined, y_train)
y_pred_lr = model_lr.predict(X_test_combined)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))


# MODELL 2: RANDOM FOREST
model_rf = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)
model_rf.fit(X_train_combined, y_train)
y_pred_rf = model_rf.predict(X_test_combined)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# VISUALISIERUNG: CONFUSION MATRIX FÜR 2 MODELLE
status_labels = sorted(df0['status'].unique())

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_lr, labels=status_labels), annot=True, fmt="d",
            ax=axes[0], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Vorhergesagt")
axes[0].set_ylabel("Tatsächlich")
axes[0].tick_params(axis='x', rotation=45)



sns.heatmap(confusion_matrix(y_test, y_pred_rf, labels=status_labels), annot=True, fmt="d",
            ax=axes[1], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Vorhergesagt")
axes[1].set_ylabel("Tatsächlich")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
# plt.savefig("../output/klassifikation_model_compar/model_comparison_confusion_matrix.png")
plt.show()

# ----------------------- Modellvisualisierung ----------------------------

# Feature-Namen für alle Spalten korrekt zusammensetzen
tfidf_feature_names = tfidf.get_feature_names_out()
all_feature_names = list(feature_cols) + list(tfidf_feature_names)

# --- LOGISTIC REGRESSION ---


# --- Heatmap Klassenkoeffizienten ---

print('\nHeatmap alle Klassen wird erstellt')
# Wir nehmen die Top 20 Features, die über alle Klassen hinweg die größte Varianz haben

# Koeffizienten-Matrix (Klassen x Features)
coef_matrix = model_lr.coef_
# Wir wählen die 20 wichtigsten Features über alle Klassen hinweg
top_features_indices = np.argsort(np.std(coef_matrix, axis=0))[-20:]

plt.figure(figsize=(12, 8))
sns.heatmap(coef_matrix[:, top_features_indices],
            yticklabels=model_lr.classes_,
            xticklabels=[all_feature_names[i] for i in top_features_indices],
            cmap="vlag", center=0, annot=False)

plt.title("Einfluss der Features pro Klasse (Logistische Regression)")
plt.xlabel("Features")
plt.ylabel("Klasse")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- RANDOM FOREST ---

print("\n--- Analyse: Random Forest ---")

# 1. Feature Namen vorbereiten
tfidf_feature_names = tfidf.get_feature_names_out()
all_feature_names = list(feature_cols) + list(tfidf_feature_names)

# 2. Feature Importance
importances = model_rf.feature_importances_
feat_imp = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis')
plt.title("Top 20 Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

