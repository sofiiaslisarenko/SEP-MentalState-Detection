import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from datenbereinigung import clean_data
from feature_builder import create_all_features, create_additional_features
from training_test import train_testdaten_split_no_stratify

nltk.download('vader_lexicon', quiet=True)  # Laedt das VADER-Lexikon herunter (nur beim ersten Mal noetig)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# VADER-Werkzeug einmal erzeugen und dann wiederverwenden
analyzer = SentimentIntensityAnalyzer()

# Daten laden und bereinigen
df0 = clean_data()

# Handgemachte Features berechnen (Pronomen-, Wort- und Satzzeichen-Zaehler)
df0 = create_additional_features(df0)
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
    'sleep_words',
    'stopwords_per_text_ratio'
]

# Alle Wort-Haeufigkeits-Spalten automatisch einsammeln
word_frequency_features = [col for col in df0.columns if col.startswith('freq_')]

# Vollstaendige Feature-Liste fuer das Modell
feature_cols = macro_features + word_frequency_features

print(f"Das Modell hat jetzt {len(feature_cols)} Features.")
print("Pruefung: Ziel darf nicht in den Features sein (muss False sein):  ",'sentiment' in feature_cols)

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
r2_score_ridge = r2_score(y_test, y_pred_ridge)
r2_train_ridge = r2_score(y_train, model_ridge.predict(X_train_scaled))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

# Modell 2: Random Forest (nicht-lineares Modell aus vielen Entscheidungsbaeumen)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
r2_score_rf = r2_score(y_test, y_pred_rf)
r2_train_rf = r2_score(y_train, model_rf.predict(X_train_scaled))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Modell 3: XGBoost (Baumbasiertes Gradient Boosting)
# learning_rate und n_estimators kontrollieren, wie fein das Modell lernt
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
model_xgb.fit(X_train_scaled, y_train)
y_pred_xgb = model_xgb.predict(X_test_scaled)
r2_score_xgb = r2_score(y_test, y_pred_xgb)
r2_train_xgb = r2_score(y_train, model_xgb.predict(X_train_scaled))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Auswertung: R2 (erklaerte Varianz), MAE (mittlerer Fehler) pro Modell und XGBoost
print("\n\n\n----------------- Auswertung -----------------\n")
print("===== Ridge =====")
print(f"Ridge: Train R² = {r2_train_ridge:.3f}, Test R² = {r2_score_ridge:.3f}")
print(f"MAE: {mae_ridge:.3f}")

print("===== Random Forest =====")
print(f"Random Forest: Train R² = {r2_train_rf:.3f}, Test R² = {r2_score_rf:.3f}")
print(f"MAE: {mae_rf:.3f}")

print("===== XGBoost =====")
print(f"XGBoost: Train R² = {r2_train_xgb:.3f}, Test R² = {r2_score_xgb:.3f}")
print(f"MAE: {mae_xgb:.3f}")


# 1. Metriken für alle Modelle berechnen (inklusive RMSE)
def evaluiere_modell(y_true, y_pred, modell_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE wird berechnet, indem wir die Wurzel (np.sqrt) aus dem Mean Squared Error ziehen
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {'Modell': modell_name, 'R²': r2, 'MAE': mae, 'RMSE': rmse}


# Ergebnisse in einer Liste sammeln
ergebnisse = [
    evaluiere_modell(y_test, y_pred_ridge, 'Ridge'),
    evaluiere_modell(y_test, y_pred_rf, 'Random Forest'),
    evaluiere_modell(y_test, y_pred_xgb, 'XGBoost')]

# In ein Pandas DataFrame umwandeln für die Visualisierung
df_ergebnisse = pd.DataFrame(ergebnisse)

# Ergebnisse in der Konsole ausgeben
print(df_ergebnisse.to_string(index=False))

# 2. Visuelles Dashboard erstellen
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Diagramm 1: Bestimmtheitsmaß R² (Höher ist besser)
sns.barplot(data=df_ergebnisse, x='Modell', y='R²', ax=ax1, palette='Blues_d')
ax1.set_title('Erklärte Varianz (R²)\n[Höher ist besser]', fontweight='bold')
ax1.set_ylim(0, max(df_ergebnisse['R²']) + 0.1)
ax1.set_ylabel('R² Score')

# Werte auf die Balken schreiben
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.3f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

# Diagramm 2: Fehlermetriken MAE und RMSE (Niedriger ist besser)
# Dafür "schmelzen" wir den DataFrame, damit Seaborn gruppierte Balken zeichnen kann
df_fehler = df_ergebnisse.melt(id_vars='Modell', value_vars=['MAE', 'RMSE'],
                               var_name='Metrik', value_name='Fehler')

sns.barplot(data=df_fehler, x='Modell', y='Fehler', hue='Metrik', ax=ax2, palette='Reds_d')
ax2.set_title('Vorhersagefehler (MAE & RMSE)\n[Niedriger ist besser]', fontweight='bold')
ax2.set_ylabel('Fehlerwert')

# Werte auf die Balken schreiben
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.3f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()


# Predicted-vs-Actual-Plot
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True, sharex=True)

predictions = [y_pred_ridge, y_pred_rf, y_pred_xgb]
names = ["Ridge", "Random Forest", "XGBoost"]
r2_scores = [r2_score_ridge, r2_score_rf, r2_score_xgb]

for ax, y_pred, name, r2 in zip(axes, predictions, names, r2_scores):
    sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red', 'label': f'{name}-Trend'})
    ax.axline((0, 0), slope=1, color='green', linestyle='--', label='Ideal: y=x')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("")
    ax.set_title(name)
    ax.text(-0.9, 0.8, f"R² = {r2:.3f}")
    ax.legend()

fig.suptitle("Vorhersage vs. VADER-Sentiment-Score")
axes[1].set_xlabel("VADER-Sentiment-Score")
axes[0].set_ylabel("Vorhergesagtes Sentiment")
plt.show()

# Residual-Plot
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True, sharex=True)

predictions = [y_pred_ridge, y_pred_rf, y_pred_xgb]
names = ["Ridge", "Random Forest", "XGBoost"]
mae = [mae_ridge, mae_rf, mae_xgb]
residuen = [(y_test - y_pred_ridge), (y_test - y_pred_rf), (y_test - y_pred_xgb)]

for ax, y_pred, name, res, maes in zip(axes, predictions, names, residuen, mae):
    sns.regplot(x=y_pred, y=res, ax=ax, scatter_kws={'alpha': 0.1}, fit_reg=False)
    ax.axhline(0, color='orange', linestyle='--', label='Null-Linie')
    ax.set_xlim(-1, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.legend()

fig.suptitle("Residual Plots")
axes[0].set_ylabel("Residuum (y_test − y_pred)")
axes[1].set_xlabel("Vorhergesagtes Sentiment")
plt.show()
































