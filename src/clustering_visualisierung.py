from datenbereinigung import clean_data
from feature_builder import create_all_features
from clustering import vorbereitung, modelle_testen, bestes_modell_waehlen

import matplotlib.pyplot as plt

def visualisierung_zwei_features(df, labels):
    df, _ = create_all_features(df)
    df['sleep_words'] = df['statement'].str.lower().str.count(
        r'\bsleep\b|\binsomnia\b|\bnight\b|\btired\b|\bawake\b'
    )
    df['cluster'] = labels

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['absolutist_count'],
        df['sleep_words'],
        c=df['cluster'],
        cmap='tab10',
        alpha=0.4,
        s=10
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Absolutistische Wörter (always, never, nothing...)')
    plt.ylabel('Schlaf-bezogene Wörter (sleep, tired, insomnia...)')
    plt.title('Clustering mit zwei Features: Absolutismus vs. Schlafprobleme')
    plt.tight_layout()
    # plt.savefig('../output/clustering/clustering_zwei_features.png')
    plt.show()
    print('Plot gespeichert als clustering_zwei_features.png')

if __name__ == "__main__":
    df = clean_data()
    X = vorbereitung(df)
    ergebnisse = modelle_testen(X)
    if ergebnisse:
        bestes, labels = bestes_modell_waehlen(ergebnisse)
        visualisierung_zwei_features(df, labels)