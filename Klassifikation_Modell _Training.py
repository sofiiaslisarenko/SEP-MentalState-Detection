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
# ============================================================
# TRAIN/TEST SPLIT FUNKTION
# Behält alle Spalten von df0, nicht nur statement/status.
# ============================================================
df0 = clean_data()
df0['word_count'] = np.random.randint(1, 500, len(df0))
df0['absolutist_ratio'] = np.random.rand(len(df0))
df0['pronoun_dominance'] = np.random.uniform(-1, 1, len(df0))

# Echte Feature-Berechnung (nicht Platzhalter):
df0['sleep_words'] = df0['statement'].str.lower().str.count(
    r'\bsleep\b|\binsomnia\b|\bnight\b|\btired\b|\bawake\b'
)
def train_testdaten_split(df0):
    """Teilt df0 in Trainings- und Testdaten auf, behält dabei alle Spalten.

    Args:
        df0 (pd.DataFrame): Der bereinigte Datensatz mit den Spalten
            'statement' und 'status'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train_df und test_df.
    """
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
# DATEN LADEN
# ============================================================
#df0 = clean_data()

# ------------------------------------------------------------
# PLATZHALTER-FEATURES
# Sobald die finalen Features vom Team feststehen, diesen Block
# durch echte Berechnungen/Merges ersetzen.
# ------------------------------------------------------------
np.random.seed(42)
df0['word_count'] = np.random.randint(1, 500, len(df0))
df0['absolutist_ratio'] = np.random.rand(len(df0))
df0['pronoun_dominance'] = np.random.uniform(-1, 1, len(df0))
# df0['sleep_words'] -> bereits vorhanden, falls schon berechnet

# ============================================================
# SPLIT
# ============================================================
train_df, test_df = train_testdaten_split(df0)

feature_cols = ['word_count', 'absolutist_ratio', 'pronoun_dominance', 'sleep_words']

X_train = train_df[feature_cols]
y_train = train_df['status']
X_test = test_df[feature_cols]
y_test = test_df['status']

# ============================================================
# SKALIERUNG
# Logistic Regression und SVM sind distanz-/gradientenbasiert,
# daher müssen die Features auf eine vergleichbare Skala gebracht werden.
# WICHTIG: fit() nur auf den Trainingsdaten, sonst Data Leakage!
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
