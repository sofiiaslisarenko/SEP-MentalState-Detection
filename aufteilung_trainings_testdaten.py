# from data_clean import clean_data


# df0 = clean_data()
from sklearn.model_selection import train_test_split
import pandas as pd

def train_testdaten_split(df0: pd.DataFrame):
    """Teilt df0 in Trainings- und Testdaten auf."""
    train_df, test_df = train_test_split(
        df0,
        test_size=0.2,
        random_state=42,
        stratify=df0['status']
    )
    print(f"Trainingsdaten: {len(train_df)} Einträge")
    print(f"Testdaten: {len(test_df)} Einträge")
    return train_df, test_df