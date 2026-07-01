import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("../output/klassifikation_model_compar", exist_ok=True)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

from datenbereinigung import clean_data
from feature_builder import create_all_features
from training_test import train_testdaten_split


# Daten laden & Feature bauen (einmal, wird von allen Experimenten geteilt)

def prepare_data():
    df0 = clean_data()
    df0, all_target_words = create_all_features(df0)
    train_df, test_df = train_testdaten_split(df0)

    macro_features = [
        'word_count',
        # 'pronoun_dominance',
        # 'absolutist_ratio',
        # 'absolute_uncertain_ratio',
        # 'time_focus_score',
        # 'future_count',
        # 'past_count',
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
    ]
    word_frequency_features = [c for c in df0.columns if c.startswith('freq_')]
    feature_cols = macro_features + word_frequency_features
    status_labels = sorted(df0['status'].unique())

    return train_df, test_df, feature_cols, status_labels

def build_custom_stopwords():
    """Englische Stoppwörter, aber Negationen & Signalwörter bewusst behalten.
    Grund: 'not', 'never' etc. tragen bei psychischen Zuständen echtes Signal
    (z.B. 'I do not feel safe') und dürfen nicht wegfallen."""
    stop = set(stopwords.words('english'))
    keep = {'no', 'not', 'nor', 'never', 'none', 'nothing', 'cannot',
            "n't", 'without', 'against'}
    return stop - keep

# Klassifikationsevaluation

def classification_evaluation(vectorizer,
                   models: dict,
                   train_df, test_df,
                   feature_cols,
                   status_labels,
                   name: str = "",
                   gewichtungs_faktor: float = 2,
                   plot_confusion: bool = False,
                   save_path: str = None) -> dict:

    """
    Trainiert & evaluiert Modelle mit gegebenem Vectorizer + numerischen Features.

    Parameter
    ---------
    vectorizer : ein (noch nicht gefitteter) TfidfVectorizer o.Ä.
                 -> die Vectorizer-Ebene passt du hier an (stop_words, ngram_range,
                    max_features, ...).
    models     : dict {name: modell-instanz}
                 -> die Modell-Ebene passt du hier an (class_weight, C, ...).
    name       : Beschriftung des Experiments (erscheint im Report & Plot-Titel).
    gewichtungs_faktor : Faktor, mit dem die (skalierten) numerischen Features
                 hochgewichtet werden, bevor sie mit TF-IDF kombiniert werden.
    plot_confusion : wenn True, wird pro Experiment eine Confusion-Matrix-Grafik
                 (ein Subplot pro Modell) erzeugt.
    save_path  : optionaler Pfad zum Speichern der Grafik (nur wenn plot_confusion).

    Rückgabe
    --------
    dict {modell-name: y_pred} – falls du die Vorhersagen weiterverwenden willst.
    """
    # y_train/y_test intern ableiten statt global
    y_train = train_df['status']
    y_test = test_df['status']

    # --- TF-IDF aus dem Statement-Text ---
    X_train_tfidf = vectorizer.fit_transform(train_df['statement'])
    X_test_tfidf = vectorizer.transform(test_df['statement'])

    # --- Numerische Features skalieren & gewichten ---
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[feature_cols]) * gewichtungs_faktor
    X_test_num = scaler.transform(test_df[feature_cols]) * gewichtungs_faktor

    # --- Kombinieren: numerische Features + TF-IDF ---
    X_train_combined = hstack([csr_matrix(X_train_num), X_train_tfidf])
    X_test_combined = hstack([csr_matrix(X_test_num), X_test_tfidf])

    # --- Training & Evaluation je Modell ---
    predictions = {}
    for model_name, model in models.items():
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        predictions[model_name] = y_pred
        print(f"\n{name} – {model_name}:\n",
              classification_report(y_test, y_pred))

    # --- Optional: Confusion Matrizen ---
    if plot_confusion:
        n = len(predictions)
        fig, axes = plt.subplots(1, n, figsize=(10 * n, 6))
        if n == 1:                     # bei einem Modell ist axes kein Array
            axes = [axes]
        for ax, (model_name, y_pred) in zip(axes, predictions.items()):
            sns.heatmap(
                confusion_matrix(y_test, y_pred, labels=status_labels),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=status_labels, yticklabels=status_labels, ax=ax,
            )
            ax.set_title(f"{model_name} – {name}")
            ax.set_xlabel("Vorhergesagt")
            ax.set_ylabel("Tatsächlich")
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        #plt.show()
        plt.close(fig)

    return predictions

def run_best_configuration():
    """Fährt die final gewählte Konfiguration: Negationen behalten,
    Bigramme, class_weight='balanced'. Das ist das Ergebnis des
    Modellvergleichs (die anderen Experimente stehen im __main__-Block)."""
    train_df, test_df, feature_cols, status_labels = prepare_data()
    custom_stop = build_custom_stopwords()

    return classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2)),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced"),
        },
        train_df=train_df, test_df=test_df,
        feature_cols=feature_cols, status_labels=status_labels,
        name="Negationen + Bigramme, class_weight balanced",
        plot_confusion=True,
    )
# Experimente (aus main.py oder direkt hier aufrufbar)
if __name__ == "__main__":
    # Ausgangspunkt aus klassifikation.py (sklearn stopwords, ohne Negationen, Unigramme)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words='english',
                                   min_df=3, max_df=0.7),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1),
        },
        name="sklearn stopwords, ohne Negationen, Unigramme",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_baseline_sklearn_sw.png",
    )

    # Beste Konfiguration (Negationen + Bigramme, class_weight balanced)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2)),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced"),
        },
        name="Negationen + Bigramme, class_weight balanced",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_bigr_balanced.png",
    )

    # Konfiguration ohne class_weight balanced
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2)),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1),
        },
        name="Negationen + Bigramme",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_bigr_rf_unbalanced.png",
    )

    # Konfiguration mit Negationen und Unigrammen (Effekt der Bigramme isolieren)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced"),
        },
        name="Negationen und Unigramme",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_unigramme.png",
    )

    # Verworfene Konfiguration mit 10k max features (Negationen + Bigramme, class_weight balanced)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=10000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2)),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced"),
        },
        name="Negationen + Bigramme, class_weight balanced, 10k max features",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_bigr_balanced_10k.png",
    )

    # Verworfene Konfiguration (Negationen + Bigramme, class_weight balanced_subsample)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2)),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced_subsample"),
        },
        name="Negationen + Bigramme, class_weight balanced_subsample",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_bigr_subsample.png",
    )

    # Verworfene Konfiguration (Negationen + Bigramme, sublinear_tf in Vectorizer)
    classification_evaluation(
        vectorizer=TfidfVectorizer(max_features=5000, stop_words=list(custom_stop),
                                   min_df=3, max_df=0.7, ngram_range=(1, 2), sublinear_tf=True),
        models={
            "Logistic Regression": LogisticRegression(max_iter=2000, C=10,
                                                      class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    n_jobs=-1, class_weight="balanced"),
        },
        name="Negationen + Bigramme, sublinear_tf in Vectorizer",
        plot_confusion=True,
        #save_path="../output/klassifikation_model_compar/model_comparison_neg_bigr_sublineartf.png",
    )

    # Weitere Experimente möglich