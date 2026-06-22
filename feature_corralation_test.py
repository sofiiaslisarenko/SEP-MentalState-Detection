from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def correlation_test(df0: pd.DataFrame):
    path = os.getcwd()
    output_path = os.path.join(path, "Output")
    # Features zum Testen
    features_to_test = [
        'pronoun_dominance', 
        'time_focus_score', 
        'absolutist_ratio', 
        'question_marks_count', 
        'word_count'
    ]

    # Wir filtern NaNs (falls z.B. durch 0 geteilt wurde) heraus und füllen sie mit 0
    df_features = df0[features_to_test + ['status']].copy()
    df_features[features_to_test] = df_features[features_to_test].fillna(0)

    # Für sauberes Plotten (damit der PC nicht abstürzt) nehmen wir eine Stichprobe, 
    # falls das Dataset größer als 2000 Zeilen ist.
    if len(df_features) > 2000:
        df_sample = df_features.sample(2000, random_state=42)
    else:
        df_sample = df_features

    # --- GRAPH 1: DER PAIRPLOT (Überschneidungen direkt sehen) ---
    print("Erstelle Pairplot (Das kann ein paar Sekunden dauern)...")
    # corner=True sorgt dafür, dass die obere Hälfte der Matrix leer bleibt (übersichtlicher)
    pair_fig = sns.pairplot(df_sample, vars=features_to_test, hue="status", corner=True, palette="husl", plot_kws={'alpha':0.5})
    pair_fig.fig.suptitle("Feature-Matrix: Wo überschneiden sich die Status?", y=1.02, fontsize=16, fontweight='bold')
    pair_fig.savefig(os.path.join(output_path, "feature_overlap_matrix.png"))
    plt.show()


    # --- GRAPH 2: t-SNE CLUSTERING (Das Upgrade für chaotische Daten) ---
    print("Berechne t-SNE Cluster-Landkarte (Dauert etwas länger als PCA)...")
    
    # 1. Ausreißer bei absoluten Zahlen stauchen (Log-Transformation)
    # np.log1p ist perfekt, weil es auch bei 0 funktioniert (log(0) wäre sonst ein Fehler)
    df_sample['word_count_log'] = np.log1p(df_sample['word_count'])
    df_sample['questions_log'] = np.log1p(df_sample['question_marks_count'])
    
    # 2. Unsere angepassten Features für das Clustering
    tsne_features = [
        'pronoun_dominance', 
        'time_focus_score', 
        'absolutist_ratio', 
        'questions_log',   # Neu: Logarithmiert
        'word_count_log'   # Neu: Logarithmiert
    ]

    # 3. Jetzt erst Skalieren
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_sample[tsne_features])

    # 4. t-SNE anwenden (Zieht organische Cluster viel besser auseinander als PCA)
    # perplexity=30 ist ein guter Standardwert für Datensätze dieser Größe
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)

    df_sample['tSNE_Dim_1'] = tsne_result[:, 0]
    df_sample['tSNE_Dim_2'] = tsne_result[:, 1]

    # Plotten
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tSNE_Dim_1", y="tSNE_Dim_2",
        hue="status",
        palette="husl",
        data=df_sample,
        legend="full",
        alpha=0.7,
        edgecolor=None
    )

    plt.title("t-SNE Cluster-Landkarte (Log-transformierte Features)", fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "tsne_cluster_map.png"))
    plt.show()

    # ------------------------------- STATISTISCHE FEATURE-EVALUATION (KRUSKAL-WALLIS) -------------------------------
    from scipy import stats
    
    print("\n--- STATISTISCHER SIGNIFIKANZ-TEST (Kruskal-Wallis) ---")
    print("Testet: Unterscheiden sich die Werte dieses Features signifikant zwischen den Status-Gruppen?")
    print("Regel: p-Wert < 0.05 bedeutet, das Feature ist EXZELLENT für das Modell.\n")

    # Wir testen unsere besten gebauten Features
    features_to_test = [
        'pronoun_dominance', 
        'time_focus_score', 
        'absolutist_ratio', 
        'question_marks_count', 
        'word_count'
    ]

    # Wir filtern NaNs heraus
    df_stats = df0[features_to_test + ['status']].copy().fillna(0)
    
    # Alle einzigartigen Status-Gruppen (Anxiety, Suicidal, Normal, etc.)
    status_groups = df_stats['status'].unique()

    # Wir loopen durch jedes Feature und machen den Test
    for feature in features_to_test:
        # Wir bereiten Listen mit den Werten für jede Gruppe vor
        # Das braucht die scipy.stats Funktion als Input
        group_data = [df_stats[df_stats['status'] == status][feature].values for status in status_groups]
        
        # Der eigentliche statistische Test
        stat, p_value = stats.kruskal(*group_data)
        
        # Ausgabe formatieren
        if p_value < 0.05:
            bewertung = "✅ SIGNIFIKANT (Feature behalten!)"
        else:
            bewertung = "❌ NICHT SIGNIFIKANT (Feature evtl. verwerfen)"
            
        print(f"Feature: {feature.upper()}")
        print(f"  Test-Statistik: {stat:.2f}")
        # Wir nutzen wissenschaftliche Notation, da p-Werte oft extrem klein sind (z.B. 0.0000001)
        print(f"  p-Wert: {p_value:.2e} -> {bewertung}\n")