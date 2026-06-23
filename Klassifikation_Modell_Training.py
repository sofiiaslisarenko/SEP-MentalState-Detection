import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_clean import clean_data
from feature_builder import create_all_features
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from aufteilung_trainings_testdaten import train_testdaten_split


# ============================================================
# DATEN LADEN & ECHTE FEATURES BERECHNEN (Hier war vorher np.random)
# ============================================================
df0 = clean_data()

df0, all_target_words = create_all_features(df0)

# ============================================================
# SPLIT & DYNAMISCHE FEATURE-LISTE
# ============================================================
train_df, test_df = train_testdaten_split(df0)

# Makro Features + das Team-Feature
macro_features = [
    'word_count', 
    'pronoun_dominance', 
    'absolutist_ratio', 
    'absolute_uncertain_ratio', 
    'time_focus_score',
    'question_marks_count',
    'ellipses_count',
    'exclamation_marks_count', 
    'sleep_words'
]

word_frequency_features = [col for col in df0.columns if col.startswith('freq_')]

feature_cols = macro_features + word_frequency_features

print(f"Das Modell hat jetzt {len(feature_cols)} Features")


X_train = train_df[feature_cols]
y_train = train_df['status']
X_test = test_df[feature_cols]
y_test = test_df['status']
# ============================================================
# TF-IDF AUS DEM STATEMENT-TEXT
# ============================================================
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(train_df['statement'])
X_test_tfidf = tfidf.transform(test_df['statement'])

# ============================================================
# SKALIERUNG DER NUMERISCHEN FEATURES
# ============================================================
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train)
X_test_num = scaler.transform(X_test)

# ============================================================
# KOMBINIEREN: NUMERISCHE FEATURES + TF-IDF
# ============================================================
X_train_combined = hstack([csr_matrix(X_train_num), X_train_tfidf])
X_test_combined = hstack([csr_matrix(X_test_num), X_test_tfidf])

# ============================================================
# MODELL 1: LOGISTIC REGRESSION
# ============================================================
model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model_lr.fit(X_train_combined, y_train)
y_pred_lr = model_lr.predict(X_test_combined)

print("===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr))

# ============================================================
# MODELL 2: SVM
# ============================================================
# model_svm = SVC(kernel="rbf", random_state=42)
# model_svm.fit(X_train_combined, y_train)
# y_pred_svm = model_svm.predict(X_test_combined)

# print("===== SVM =====")
# print(classification_report(y_test, y_pred_svm))

# ============================================================
# MODELL 3: RANDOM FOREST
# ============================================================
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)
model_rf.fit(X_train_combined, y_train)
y_pred_rf = model_rf.predict(X_test_combined)

print("===== Random Forest =====")
print(classification_report(y_test, y_pred_rf))

# ============================================================
# VISUALISIERUNG: CONFUSION MATRIX FÜR ALLE 3 MODELLE
# ============================================================
status_labels = sorted(df0['status'].unique())

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_lr, labels=status_labels), annot=True, fmt="d",
            ax=axes[0], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Vorhergesagt")
axes[0].set_ylabel("Tatsächlich")
axes[0].tick_params(axis='x', rotation=45)

# sns.heatmap(confusion_matrix(y_test, y_pred_svm, labels=status_labels), annot=True, fmt="d",
#             ax=axes[1], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
# axes[1].set_title("SVM")
# axes[1].set_xlabel("Vorhergesagt")
# axes[1].set_ylabel("Tatsächlich")
# axes[1].tick_params(axis='x', rotation=45)

sns.heatmap(confusion_matrix(y_test, y_pred_rf, labels=status_labels), annot=True, fmt="d",
            ax=axes[1], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Vorhergesagt")
axes[1].set_ylabel("Tatsächlich")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("model_comparison_confusion_matrix.png")
plt.show()