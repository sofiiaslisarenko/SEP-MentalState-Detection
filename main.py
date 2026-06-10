#Main Code
import Pandas as pd
import glob
#Dataset import von Kaggle (von Kaggle selbst so vorgeschlagen)
import kagglehub
import os
# Download latest version
file_paths = [kagglehub.dataset_download("nikhileswarkomati/suicide-watch"),            # Suicide and Depression Detection
              kagglehub.dataset_download("szegeelim/mental-health"),                    # Depression Detection using Sentiment Analysis
              kagglehub.dataset_download("infamouscoder/depression-reddit-cleaned")     #Depression: Reddit Dataset (Cleaned)
]
for path in file_paths:
    print("=" * 80)
    print("FILE:", os.path.basename(path))
    print("Path to dataset files:", path)

    df = pd.read_csv(path)

    # Show columns first
    print("\nColumns:", df.columns.tolist())

    # Try to show value counts for ALL text-like columns
    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        print(f"\n🔹 Value counts for column: '{col}'")
        print(df[col].value_counts())

pd.read_csv