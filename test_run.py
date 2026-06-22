
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
    
def test_run(df0 : pd.DataFrame):
    print("\n--- TRAINIERE BASELINE-MODELL (Nur mit Meta-Features) ---")

    # 1. Daten vorbereiten (Unsere bewiesenen Gewinner-Features aus dem Kruskal-Wallis Test)
    features_for_model = [
        'pronoun_dominance', 
        'time_focus_score', 
        'absolutist_ratio', 
        'question_marks_count', 
        'word_count'
    ]

    # X sind unsere Daten (Features), y ist unser Ziel (Die Diagnosen/Status)
    X = df0[features_for_model].fillna(0)
    y = df0['status']

    # 2. Train-Test-Split (Die goldene Regel!)
    # Wir verstecken 20% der Daten vor der KI, um sie später fair zu testen.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Das Modell initialisieren
    # 100 Entscheidungsbäume (n_estimators), die zusammen abstimmen
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 4. Das Modell trainieren (Hier lernt die KI die psychologischen Muster)
    print("Modell lernt die psychologischen Muster... (Das geht sehr schnell)")
    rf_model.fit(X_train, y_train)

    # 5. Die KI auf die ungesehenen 20% loslassen
    y_pred = rf_model.predict(X_test)

    # 6. Auswertung: Wie gut ist das Modell?
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Genauigkeit (Accuracy) auf ungesehenen Daten: {accuracy * 100:.2f}%\n")
    print("Detaillierter Report (Wie gut wird jeder Status erkannt?):")
    # Der Report zeigt dir Precision (Wie oft lag die KI richtig, wenn sie "Suicidal" geraten hat?) 
    # und Recall (Wie viele der echten "Suicidal"-Texte hat sie gefunden?)
    print(classification_report(y_test, y_pred))


    # ------------------------------- EINBLICK IN DAS KI-GEHIRN (Die Prozent-Skala) -------------------------------
    print("\n--- WAHRSCHEINLICHKEITEN (PROBABILITIES) FÜR EINZELNE TEXTE ---")
    
    # Wir nehmen einfach mal die ersten 3 Texte aus unserem versteckten Test-Set
    beispiel_texte = X_test.head(3)
    echte_labels = y_test.head(3).values

    # predict_proba ist die magische Funktion, die dir die 0 bis 1 Skala gibt
    wahrscheinlichkeiten = rf_model.predict_proba(beispiel_texte)
    klassen_namen = rf_model.classes_

    for i in range(len(beispiel_texte)):
        print(f"\nText {i+1} (Wahre Diagnose laut Datensatz: {echte_labels[i]})")
        print("Das Modell ist sich zu ... sicher:")
        
        # Wir sortieren die Vorhersagen, damit die höchste Prozentzahl oben steht
        # Das zip verbindet die Klassen-Namen mit den Prozentzahlen
        vorhersage_liste = list(zip(klassen_namen, wahrscheinlichkeiten[i]))
        vorhersage_liste.sort(key=lambda x: x[1], reverse=True)
        
        for klasse, prozent in vorhersage_liste:
            # Wir runden auf 2 Nachkommastellen für eine schöne 0.00 bis 1.00 Skala
            print(f"  - {klasse}: {prozent:.2f} ({prozent * 100:.1f}%)")