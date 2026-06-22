import pandas as pd                 #Für das Erstellen und Arbeiten mit Dataframes
import os                           #Zum finden des Paths (File Ordner)
import matplotlib.pyplot as plt     #Zusammen mit seaborn zum erstellen von Graphen
import re                           #Für Regex Operationen
import seaborn as sns

path = os.getcwd()
output_path = os.path.join(path, "Output")
# Sebastians Explorative Daten analyse:
def expl_data(df0 : pd.DataFrame):
    """Graphische Darstellung der Daten zum Visuellen analysieren."""

    # ändere den path auf den outputfolder:
    os.chdir(output_path)

    #BoW u token per message...
    #sentence len
    #pronouns
    #Pronouns:
    #remove the largest suicidal statement
    #This is done to remove an outlier that would skew the data
    # c = df0[df0['status'] == 'Suicidal']['statement'].str.len().idxmax()
    # df0.drop(index=c, inplace=True)
    # Das ?i sagt dem Programm das groß und klein schreibung egal ist. r steht für raw string, damit kann man backslashes \ in den String schreiben ohne dass das folge Zeichen excaped
    self_pronouns = r'(?i)\b(we|us|ourselves|ourself|i|me|mine|myself|my)\b'
    other_pronouns = r'(?i)\b(you|your|yours|yourself|yourselves|he|she|her|hers|herself|him|his|himself|they|them|their|theirs|themselves|thyself|thine)\b'



    # -------------------------------------------- ZÄHLEN ----------------------------------------------
    # Zählen:
    #selbstbezogene Pronomen in Statement
    df0['self_pronouns_count'] = df0["statement"].str.count(self_pronouns)
    #alle Pronomen in Statement
    df0['all_pronouns_count'] = df0["statement"].str.count(other_pronouns) + df0['self_pronouns_count']
    # Alle nicht selbstbezogenen Pronomen in Statement, ausgenommen Objektbezogene wie "it"
    df0['other_pronouns_count'] = df0["statement"].str.count(other_pronouns)
    # total word count in Statement
    df0['word_count'] = df0["statement"].str.split().str.len()
    # wie oft werden ich bezogene Pronomen im verhältniss zur länge der Nachricht verwendet
    df0['self_pronoun_to_word_ratio'] = df0['self_pronouns_count'] / df0['word_count']
    # die Differenz der selbstbezogenen Pronomen und der anderen Pronomen 
    df0['dif_pronouns'] = df0['self_pronouns_count'] - df0['other_pronouns_count']
    # Normalisieren der Differenz, bzw wie dominant sind die selbstbezogenen Pronomen auf einer Skala von -1 bis 1
    df0['pronoun_dominance'] = ((df0['self_pronouns_count'] - df0['other_pronouns_count']) / df0['all_pronouns_count']).fillna(0)   #Fillna ist dazu da um rechenfehler bzw na's bei 0 Pronomen zu vermeiden

    df0.info()
    print(df0.head())


    # -------------------------------------------- ANALYSEN ----------------------------------------------
    # Analysieren von summen, abweichungen, anteilen und Durchschnitten/median
    anteil_self_to_all = df0.groupby('status')['self_pronouns_count'].sum() / df0.groupby('status')['all_pronouns_count'].sum() #sum summiert die gruppierten einträge von df0 status
    print(f"Anteil der selbstbezogene Pronomen (plural + singular) in allen Pronomen: \n{anteil_self_to_all}")

    std_self = df0.groupby('status')['self_pronouns_count'].std()   #std() ist die funktion für das einfache berechnen der standart abweichung
    print(f"Standardabweichung des Vorkommens von selbstbezogenen Pronomen (plural + singular) pro Gruppe in statements: \n{std_self}")
    
    std_all = df0.groupby('status')['all_pronouns_count'].std()
    print(f"Standardabweichung aller Pronomen pro Gruppe in statements: \n{std_all}")
    
    std_dif_pron = df0.groupby('status')['dif_pronouns'].std()
    print(f"Standartabweichung der Differenz der selbstb. Pronomen und der anderen: \n{std_dif_pron}")
    
    x_len = df0.groupby('status')['word_count'].median()
    std_len = df0.groupby('status')['word_count'].std()
    print(f"Median der Wortlänge jeder Gruppe und ihre Standardabweichung: \n{x_len}\n{std_len}")
    
    std_pronoun_ratio = df0.groupby('status')['self_pronoun_to_word_ratio'].std()
    durchschnitt = df0.groupby('status')['self_pronoun_to_word_ratio'].mean()
    median = df0.groupby('status')['self_pronoun_to_word_ratio'].median()
    print(f"Durchschnitt der self pronouns to words Ratio pro Gruppe: \n{durchschnitt}")
    print(f"Median der self pronouns to words Ratio pro Gruppe: \n{median}")
    
    print(f"Standardabweichung der self pronouns to words Ratio pro Gruppe: \n{std_pronoun_ratio}")
    x_dif_pron = df0.groupby('status')['dif_pronouns'].mean()
    print(f"Median der Differenz der Pronomen pro Gruppe: \n{x_dif_pron}")
    


    #------------------------------------------------- GRAPHEN ------------------------------------------------------
    # --- Erster Graph ---
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.set_theme(style="whitegrid", palette="husl")
    fig.set_size_inches(16, 9)
    fig.suptitle("Pronomen-Analyse: Anteil und Verteilung", fontsize=18, fontweight='bold', y=0.98)
    
    sns.barplot(ax=axes[0], x=anteil_self_to_all.index, y=anteil_self_to_all.values, hue=anteil_self_to_all.index, legend=True)
    axes[0].set_title("Anteil der Ich-Pronomen an allen Pronomen", fontsize=14)
    axes[0].set_ylabel("Ratio in %")
    axes[0].set_xlabel("Status")
    axes[0].set_yticklabels([f'{x*100:.0f}%' for x in axes[0].get_yticks()])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    sns.boxenplot(ax=axes[1], x="self_pronoun_to_word_ratio", y="status", data=df0, hue="status")
    axes[1].set_title("Verhältnis: Ich-Pronomen zur Textlänge", fontsize=14)
    axes[1].set_xlabel("Ratio (Ich-Pronomen / Gesamtwörter)")
    axes[1].set_ylabel("") # Status steht schon an der Achse
    
    plt.tight_layout() 
    fig.savefig("pronoun_analysis_1.png") # Speichern vor show()
    plt.show()

    # --- Zweiter Graph ---
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.set_theme(style="whitegrid", palette="husl") # Theme nochmal für diese Figur anwenden
    fig.set_size_inches(16, 9)
    fig.suptitle("Verteilung der Text-Features", fontsize=18, fontweight='bold', y=0.98)
    
    sns.histplot(ax=axes[0,0], data=df0, x="self_pronoun_to_word_ratio", hue="status", kde=True, bins=40)
    axes[0,0].set_title("Dichteverteilung: Ich-Pronomen-Ratio", fontsize=12)
    axes[0,0].set_xlabel("Ratio (Ich-Pronomen / Gesamtwörter)")
    
    sns.violinplot(ax=axes[0,1], data=df0, y="pronoun_dominance", x="status", hue="status")   
    axes[0,1].set_title("Pronomen-Dominanz", fontsize=12)
    axes[0,1].set_ylabel("Score (-1 = Andere, +1 = Selbst)")
    axes[0,1].set_xlabel("Status")
    
    sns.histplot(ax=axes[1,0], data=df0, x="word_count", hue="status", kde=True, log_scale=True)
    axes[1,0].set_title("Verteilung der Textlängen", fontsize=12)
    axes[1,0].set_xlabel("Anzahl Wörter (Logarithmisch)")
    
    sns.histplot(ax=axes[1,1], data=df0, x="all_pronouns_count", hue="status", kde=False, log_scale=True, bins=30)
    axes[1,1].set_title("Verteilung aller genutzten Pronomen", fontsize=12)
    axes[1,1].set_xlabel("Anzahl Pronomen (Logarithmisch)")
    
    plt.tight_layout()
    fig.savefig("pronoun_analysis_2.png")
    plt.show()



    # ------------------------------- WEITERE ANALYSEN (VORBEREITUNG) -------------------------------
    # Absolutistische Wörter (Alles oder Nichts Denken)
    absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all)\b'
    
    # Zeitliche Orientierung (Past und Future)
    # Erstmal wenige standart wörter
    past_words = r'(?i)\b(was|were|had|did|been|could)\b'
    future_words = r'(?i)\b(will|shall|going to|might|worry|worried|anxious)\b'
    


    # ------------------------------- WEITERE ANALYSEN (ZÄHLEN) -------------------------------------
    # Absolutismus-Quote (Absolutistische Wörter pro Wort)
    df0['absolutist_count'] = df0["statement"].str.count(absolutist_words)
    df0['absolutist_ratio'] = df0['absolutist_count'] / df0['word_count']
    
    # Zeit-Fokus (-1 = Komplett Zukunft, +1 = Komplett Vergangenheit)
    df0['past_count'] = df0["statement"].str.count(past_words)
    df0['future_count'] = df0["statement"].str.count(future_words)
    df0['total_time_words'] = df0['past_count'] + df0['future_count']
    df0['time_focus_score'] = ((df0['past_count'] - df0['future_count']) / df0['total_time_words']).fillna(0)
    
    # Zeichensetzung als Emotions-Barometer (Fragezeichen und Resignation/Pausen)
    # Der Backslash \ ist wichtig, weil ? und . in Regex Sonderzeichen sind.
    df0['question_marks_count'] = df0["statement"].str.count(r'\?')
    df0['ellipses_count'] = df0["statement"].str.count(r'\.\.\.')
    df0['exclamation_marks_count'] = df0["statement"].str.count(r'\!')
    


    # ------------------------------- WEITERE ANALYSEN (AUSWERTUNG) -------------------------------------
    # auswertung
    print("\n------")
    durchschnitt_absolutist = df0.groupby('status')['absolutist_ratio'].mean()
    print(f"Durchschnittliche Absolutismus-Quote pro Gruppe: \n{durchschnitt_absolutist}")
    
    durchschnitt_time = df0.groupby('status')['time_focus_score'].mean()
    print(f"Zeit-Fokus (-1=Zukunft, +1=Vergangenheit) pro Gruppe: \n{durchschnitt_time}")
    
    durchschnitt_fragen = df0.groupby('status')['question_marks_count'].mean()
    print(f"Durchschnittliche Fragezeichen pro Statement: \n{durchschnitt_fragen}")
    
    durchschnitt_ellipsen = df0.groupby('status')['ellipses_count'].mean()
    print(f"Durchschnittliche Ellipsen (...) pro Statement: \n{durchschnitt_ellipsen}")

    durchschnitt_ausrufezeichen = df0.groupby('status')['exclamation_marks_count'].mean()
    print(f"Durchschnittliche Ausrufezeichen (!) pro Statement: \n{durchschnitt_ausrufezeichen}")


    # 2. Eine große, aufgeräumte Figur mit 4 Bereichen (2x2) erstellen
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    fig.suptitle("Psychologische Textprofile im Vergleich", fontsize=20, fontweight='bold', y=0.98)

    # --- Graph 1: Ausrufezeichen (Oben Links) ---
    sns.barplot(ax=axes[0, 0], data=df0, x="exclamation_marks_count",y="status", hue="status")
    axes[0, 0].set_title("Verteilung der Ausrufezeichen", fontsize=14)
    axes[0, 0].set_xlabel("Anzahl Ausrufezeichen pro Statement")
    axes[0, 0].set_ylabel("")

    # Wir multiplizieren die Ratio mit 100, um schöne Prozentzahlen (0-2%) auf der Achse zu haben
    sns.barplot(ax=axes[0, 1], data=df0, x="ellipses_count", y="status", hue="status", legend=False)
    axes[0, 1].set_title("Durchschnittliche Ellipsen (...) pro Statement", fontsize=14)
    axes[0, 1].set_xlabel("Anzahl Ellipsen (...)")
    axes[0, 1].set_ylabel("")

    # --- Graph 3: Fragezeichen-Dichte (Unten Links) ---
    # Wir nehmen den Durchschnitt (mean) per Barplot. Die kleinen Striche sind die Fehlertoleranz.
    sns.barplot(ax=axes[1, 0], data=df0, x="question_marks_count", y="status", hue="status", legend=False)
    axes[1, 0].set_title("Durchschnittliche Fragen pro Statement", fontsize=14)
    axes[1, 0].set_xlabel("Anzahl Fragezeichen")
    axes[1, 0].set_ylabel("")

    # --- Graph 4: Absolutismus-Quote (Unten Rechts) ---
    # Wir multiplizieren die Ratio mit 100, um schöne Prozentzahlen (0-2%) auf der Achse zu haben
    df0['absolutist_percent'] = df0['absolutist_ratio'] * 100
    sns.barplot(ax=axes[1, 1], data=df0, x="absolutist_percent", y="status", hue="status", legend=False)
    axes[1, 1].set_title("Anteil absolutistischer Wörter", fontsize=14)
    axes[1, 1].set_xlabel("Anteil am Gesamttext in %")
    axes[1, 1].set_ylabel("")

    # 3. Das Layout atmen lassen (Abstände anpassen)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("pronoun_analysis_3.png")
    plt.show()

    # --- Graph: Zeit-Fokus (Oben Rechts) ---
    fig = plt.figure(figsize=(16, 9))
    axes = fig.add_axes((0.1, 0.1, 0.8, 0.8)) # Manuelle Achsenpositionierung
    sns.violinplot(data=df0, x="time_focus_score", y="status", hue="status", legend=False)
    axes.axvline(0, color='red', linestyle='--', alpha=0.6) # Rote Null-Linie
    axes.set_title("Fokus: Zukunft vs. Vergangenheit", fontsize=14)
    axes.set_xlabel("<- Zukunft / Sorgen (0.0) Vergangenheit / Reue ->")
    axes.set_ylabel("")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("pronoun_analysis_4.png")
    plt.show()
    
    #----------------------------------- GEWONNENE DATEN ALS CSV SPEICHERN --------------------------------
    data_stored = [anteil_self_to_all,  x_dif_pron, x_len, std_self, std_all, std_dif_pron,durchschnitt_absolutist, median, durchschnitt_time, durchschnitt, durchschnitt_fragen, durchschnitt_ellipsen, durchschnitt_ausrufezeichen]
    data_stored_names = ['anteil_self_to_all', 'x_dif_pron', 'x_len', 'std_self', 'std_all', 'std_dif_pron', 'durchschnitt_absolutist', 'self_pronoun_to_word_ratio_median', 'durchschnitt_time', 'self_pronoun_to_word_ratio', 'durchschnitt_fragen', 'durchschnitt_ellipsen', 'durchschnitt_ausrufezeichen']
    df_results = pd.DataFrame(data_stored, index=data_stored_names)
    df_results.to_csv('analysis_results.csv')

    #return df0[]