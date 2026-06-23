import pandas as pd
import numpy as np
from data_clean import clean_data
from feature_builder import create_all_features
# Verwandlung von Texte in Zahlen
from sklearn.feature_extraction.text import TfidfVectorizer
#Text-Features und Zahlen-Features zusammen verwenden(horizontal stack, csr_matrix- speichersparendes Format)
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD  # TF-IDF-Matrix verkleinern
from sklearn.preprocessing import normalize, StandardScaler # Werte vereinheitlichen

# === Clustering Methode ===
from sklearn.metrix import KMeans, DBSCAN, AgglomerativeClustering
# === Qualitätsbewertung von Modellen ===
from sklearn.metrics import silhouette_score, davies_bouldin_score
# === Visualisation ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
