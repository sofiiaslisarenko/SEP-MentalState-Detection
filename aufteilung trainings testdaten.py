from data_clean import clean_data


df0 = clean_data()
def train_testdaten_split(df0):

    X = df0['text']
    y = df0['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Trainingsdaten: {len(X_train)} Einträge")
    print(f"Testdaten: {len(X_test)} Einträge")

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    #train_df.to_csv('train.csv', index=False)
    #test_df.to_csv('test.csv', index=False)
    return train_df, test_df
    print("Gespeichert: train.csv und test.csv")