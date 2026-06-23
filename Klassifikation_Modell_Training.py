import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_clean import clean_data
from feature_builder import create_all_features 

# ============================================================
# DATEN LADEN & ECHTE FEATURES BERECHNEN (Hier war vorher np.random)
# ============================================================
df0 = clean_data()

df0, all_target_words = create_all_features(df0)

df0['sleep_words'] = df0['statement'].str.lower().str.count(
    r'\bsleep\b|\binsomnia\b|\bnight\b|\btired\b|\bawake\b'
)


# ============================================================
# TRAIN/TEST SPLIT FUNKTION
# Behält alle Spalten von df0, nicht nur statement/status.
# ============================================================
def train_testdaten_split(df0 : pd.DataFrame):
    """Teilt df0 in Trainings- und Testdaten auf, behält dabei alle Spalten."""
    train_df, test_df = train_test_split(
        df0,
        test_size=0.2,
        random_state=42,
        stratify=df0['status']
    )

    print(f"Trainingsdaten: {len(train_df)} Einträge")
    print(f"Testdaten: {len(test_df)} Einträge")

    return train_df, test_df


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
# SKALIERUNG
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODELL 1: LOGISTIC REGRESSION
# ============================================================
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)

print("===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr))

# ============================================================
# MODELL 2: SVM
# ============================================================
model_svm = SVC(kernel="rbf", random_state=42)
model_svm.fit(X_train_scaled, y_train)
y_pred_svm = model_svm.predict(X_test_scaled)

print("===== SVM =====")
print(classification_report(y_test, y_pred_svm))

# ============================================================
# VISUALISIERUNG: CONFUSION MATRIX FÜR BEIDE MODELLE
# ============================================================
status_labels = sorted(df0['status'].unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_lr, labels=status_labels), annot=True, fmt="d",
            ax=axes[0], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Vorhergesagt")
axes[0].set_ylabel("Tatsächlich")
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(confusion_matrix(y_test, y_pred_svm, labels=status_labels), annot=True, fmt="d",
            ax=axes[1], cmap="Blues", xticklabels=status_labels, yticklabels=status_labels)
axes[1].set_title("SVM")
axes[1].set_xlabel("Vorhergesagt")
axes[1].set_ylabel("Tatsächlich")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("model_comparison_confusion_matrix.png")
plt.show()