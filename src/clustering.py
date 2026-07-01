from datenbereinigung import clean_data
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def vorbereitung(df):
    """Text in Zahlen umwandeln und Dimensionen reduzieren"""
    print("=== Vorbereitung der Daten ===")

    # Text → TF-IDF Matrix
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(df["statement"])
    print(f"TF-IDF Matrix Größe: {X.shape}")

    # Dimensionsreduktion (wichtig für Clustering auf Text)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = normalize(svd.fit_transform(X))
    print(f"Nach Reduktion: {X_reduced.shape}")

    return X_reduced

def modelle_testen(X):
    """Verschiedene Clustering-Modelle testen und vergleichen"""
    print("\n=== Modelle testen ===")

    modelle = {
        "KMeans (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
        "KMeans (k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
        "KMeans (k=7)": KMeans(n_clusters=7, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=10) }

    ergebnisse = {}
    for name, modell in modelle.items():
        labels = modell.fit_predict(X)
        anzahl_cluster = len(set(labels)) - (1 if -1 in labels else 0)

        if anzahl_cluster > 1:
            silhouette = silhouette_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            ergebnisse[name] = {
                "labels": labels,
                "silhouette": silhouette,
                "davies": davies,
                "cluster": anzahl_cluster,
            }
            print(f"{name}: Silhouette={silhouette:.3f} | Davies-Bouldin={davies:.3f} | Cluster={anzahl_cluster}")
        else:
            print(f"{name}: Kein sinnvolles Clustering gefunden")

    return ergebnisse
def bestes_modell_waehlen(ergebnisse):
    """Bestes Modell anhand Silhouette Score auswählen"""
    print("\n=== Bestes Modell ===")

    # Höherer Silhouette Score = besseres Clustering
    bestes = max(ergebnisse, key=lambda x: ergebnisse[x]["silhouette"])
    print(f"✅ Bestes Modell: {bestes}")
    print(f"   Silhouette Score: {ergebnisse[bestes]['silhouette']:.3f}")
    print(f"   Davies-Bouldin:   {ergebnisse[bestes]['davies']:.3f}")
    print(f"   Anzahl Cluster:   {ergebnisse[bestes]['cluster']}")

    return bestes, ergebnisse[bestes]["labels"]
def visualisierung(X, labels, modellname):
    """Cluster in 2D darstellen"""
    print("\n=== Visualisierung ===")

    # Auf 2D reduzieren für Plot
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Clustering: {modellname}")
    plt.xlabel("PCA Komponente 1")
    plt.ylabel("PCA Komponente 2")
    plt.tight_layout()
    plt.savefig("../output/clustering/clustering_ergebnis.png")
    plt.show()
    plt.close()
    print("Plot gespeichert als clustering_ergebnis.png")

def heatmap_cluster_vs_status(df):
    """Heatmap der Cluster-Verteilung im Vergleich zu echten Klassen"""
    comparison = df.groupby(['cluster', 'status']).size().unstack(fill_value=0)
    pct = comparison.div(comparison.sum(axis=1), axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pct, annot=True, fmt=".0%", cmap="Blues",
                linewidths=0.5, cbar_kws={'label': 'Anteil im Cluster'})
    plt.title("Cluster vs. echte Klassen (in Prozent)")
    plt.xlabel("Klasse")
    plt.ylabel("Cluster")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("../output/clustering/cluster_vs_status_heatmap.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    df = clean_data()
    X = vorbereitung(df)
    ergebnisse = modelle_testen(X)
    if ergebnisse:
        bestes, labels = bestes_modell_waehlen(ergebnisse)

        visualisierung(X, labels, bestes)

        # Labels zum DataFrame hinzufügen
        df["cluster"] = labels
        print("\n Cluster vs Echte Klassen:")
        comparison = df.groupby(["cluster", "status"]).size().unstack(fill_value=0)
        print(comparison)
        print("\n In Prozent:")
        print(comparison.div(comparison.sum(axis=1), axis=0).round(2))
        print("\nCluster-Verteilung:")
        print(df["cluster"].value_counts())
        heatmap_cluster_vs_status(df)
    else:
        print("Kein Modell hat funktioniert. Parameter anpassen.")