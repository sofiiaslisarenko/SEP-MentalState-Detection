import pandas as pd                 #Für das Erstellen und Arbeiten mit Dataframes
import os                           #Zum finden des Paths (File Ordner)
import matplotlib.pyplot as plt     #Zusammen mit seaborn zum erstellen von Graphen
import re                           #Für Regex Operationen
import seaborn as sns


def expl_data(df0 : pd.DataFrame):
    """Graphische Darstellung der Daten zum Visuellen analysieren."""

    # ändere den path auf den outputfolder:
    
    path = os.getcwd()
    output_path = os.path.join(path, "Output")
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
    words_over = df0['word_count'] > 10
    df0 = df0[words_over]
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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (16,9))
    sns.set_theme(style="whitegrid", palette="husl") # Theme nochmal für diese Figur anwenden
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
    # Absolutistische Wörter
    absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all|nobody|forever|ever|noone|everyone|everybody|i know|impossible|must)\b'
    uncertain_words = r'(?i)\b(maby|perhaps|possibly|possible|may|might|could|i think|not sure|uncertain)\b'
    # Zeitliche Orientierung
    past_words = r'(?i)\b(was|were|had|did|been|could|said|went|ago|last|got|wanted|used|liked)\b'
    future_words = r"(?i)\b(will|shall|going to|might|worry|worried|anxious|'ll)\b"
    


    # ------------------------------- WEITERE ANALYSEN (ZÄHLEN) -------------------------------------
    # Absolutismus-Quote (Absolutistische Wörter pro Wort)
    df0['absolutist_count'] = df0["statement"].str.count(absolutist_words)
    df0['absolutist_ratio'] = df0['absolutist_count'] / df0['word_count']
    df0['uncertain_count'] = df0["statement"].str.count(uncertain_words)
    df0['uncertain_ratio'] = df0['uncertain_count'] / df0['word_count']
    df0['absolute_uncertain_ratio'] = (df0['absolutist_count'] - df0['uncertain_count']) / (df0['uncertain_count'] + df0['absolutist_count']).fillna(0)
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


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    fig.suptitle("Psychologische Textprofile im Vergleich", fontsize=20, fontweight='bold', y=0.98)

    # ---  Ausrufezeichen  ---
    sns.barplot(ax=axes[0, 0], data=df0, x="exclamation_marks_count",y="status", hue="status")
    axes[0, 0].set_title("Verteilung der Ausrufezeichen", fontsize=14)
    axes[0, 0].set_xlabel("Anzahl Ausrufezeichen pro Statement")
    axes[0, 0].set_ylabel("")


    sns.barplot(ax=axes[0, 1], data=df0, x="ellipses_count", y="status", hue="status", legend=False)
    axes[0, 1].set_title("Durchschnittliche Ellipsen (...) pro Statement", fontsize=14)
    axes[0, 1].set_xlabel("Anzahl Ellipsen (...)")
    axes[0, 1].set_ylabel("")

    # ---Fragezeichen-Dichte ---
    # Wir nehmen den Durchschnitt (mean) per Barplot. Die kleinen Striche sind die Fehlertoleranz.
    sns.barplot(ax=axes[1, 0], data=df0, x="question_marks_count", y="status", hue="status", legend=False)
    axes[1, 0].set_title("Durchschnittliche Fragen pro Statement", fontsize=14)
    axes[1, 0].set_xlabel("Anzahl Fragezeichen")
    axes[1, 0].set_ylabel("")

    # --- Absolutismus-Quote ---
    # Wir multiplizieren die Ratio mit 100, um schöne Prozentzahlen (0-2%) auf der Achse zu haben
    df0['absolutist_percent'] = df0['absolutist_ratio'] * 100
    sns.barplot(ax=axes[1, 1], data=df0, x="absolutist_percent", y="status", hue="status", legend=False)
    axes[1, 1].set_title("Anteil absolutistischer Wörter", fontsize=14)
    axes[1, 1].set_xlabel("Anteil am Gesamttext in %")
    axes[1, 1].set_ylabel("")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("pronoun_analysis_3.png")
    plt.show()

    # --- Zeit-Fokus ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (16,9)) # Manuelle Achsenpositionierung
    sns.violinplot(ax = axes[1],data=df0, x="time_focus_score", y="status", hue="status", legend=False)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.6) # Rote Null-Linie
    axes[1].set_title("Fokus: Zukunft vs. Vergangenheit", fontsize=14)
    axes[1].set_xlabel("<- Zukunft / Sorgen (0.0) Vergangenheit / Reue ->")
    axes[1].set_ylabel("")

    # sns.clustermap(ax = axes[0], data = df0, x = "absolutist_ratio", y = "uncertain_ratio", hue = "status", legend= False)
    # axes[0].axvline(0, color='red', linestyle='--', alpha=0.6)
    # plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("pronoun_analysis_4.png")
    plt.show()
    
    #----------------------------------- DATEN ALS CSV SPEICHERN --------------------------------
    data_stored = [anteil_self_to_all,  x_dif_pron, x_len, std_self, std_all, std_dif_pron,durchschnitt_absolutist, median, durchschnitt_time, durchschnitt, durchschnitt_fragen, durchschnitt_ellipsen, durchschnitt_ausrufezeichen]
    data_stored_names = ['anteil_self_to_all', 'x_dif_pron', 'x_len', 'std_self', 'std_all', 'std_dif_pron', 'durchschnitt_absolutist', 'self_pronoun_to_word_ratio_median', 'durchschnitt_time', 'self_pronoun_to_word_ratio', 'durchschnitt_fragen', 'durchschnitt_ellipsen', 'durchschnitt_ausrufezeichen']
    df_results = pd.DataFrame(data_stored, index=data_stored_names)
    df_results.to_csv('analysis_results.csv')

    os.chdir(path)
    return df0




# ----------------------------------------------------------------------------------------------------


def absolute_uncertain(df0 : pd.DataFrame, statement_len_filter = 0):
    """
    absolutist uncertain und negative wörter werden pro statement gezählt und als durchschnitt pro Gruppe ausgegeben und als Tabelle ausgegeben.
    \nstatement_len_filter sortiert alle statements aus, deren wörter länge kleiner als der übergebene int ist. (Default = 0)
    \nreturnt ein dict.
    """
    path = os.getcwd()
    output_path = os.path.join(path, "Output")
    os.chdir(output_path)
    min_statement_len = statement_len_filter #!!!!ACHTUNG aktuell werden nur statements mit dieser Anzahl genommen und alle anderen aussortiert!!!!
    df0['word_count'] = df0["statement"].str.split().str.len()
    words_over = df0['word_count'] > min_statement_len
    df0 = df0[words_over]
    print(df0["status"].value_counts())
    #output_path = os.path.join(path, "Output")
    #os.chdir(output_path)
    # Absolutistische Wörter (Alles oder Nichts Denken)
    # absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all|nobody|forever|ever|noone|everyone|everybody|i know|impossible|must)\b'
    # uncertain_words = r'(?i)\b(maybe|perhaps|possibly|possible|may|might|could|i think|not sure|uncertain)\b'
    # Zeitliche Orientierung (Past und Future)
    # past_words = r'(?i)\b(was|were|had|did|been|could|said|went|ago|last|got|wanted|used|liked)\b'
    # future_words = r"(?i)\b(will|shall|going to|might|worry|worried|anxious|'ll)\b"
    # Zeit-Fokus (-1 = Komplett Zukunft, +1 = Komplett Vergangenheit)
    # df0['past_count'] = df0["statement"].str.count(past_words)
    # df0['future_count'] = df0["statement"].str.count(future_words)
    # df0['total_time_words'] = df0['past_count'] + df0['future_count']
    # df0['time_focus_score'] = ((df0['past_count'] - df0['future_count']) / df0['total_time_words']).fillna(0)
    # Absolutismus-Quote (Absolutistische Wörter pro Wort)
    # df0['word_count'] = df0["statement"].str.split().str.len()
    # words_over = df0['word_count'] > 10
    # df0 = df0[words_over]
    # df0['absolutist_count'] = df0["statement"].str.count(absolutist_words)
    # df0['absolutist_ratio'] = df0['absolutist_count'] / df0['word_count']
    # df0['uncertain_count'] = df0["statement"].str.count(uncertain_words)
    # df0['uncertain_ratio'] = df0['uncertain_count'] / df0['word_count']
    # df0['absolute_uncertain_ratio'] = ((df0['absolutist_count'] - df0['uncertain_count']) / (df0['uncertain_count'] + df0['absolutist_count'])).fillna(0)
    # maske_echte_werte = df0[['total_time_words','uncertain_count','absolutist_count']].any(axis = 1)
    # df_plot = df0[maske_echte_werte]
    # uncertain_mean = df0.groupby("status")["uncertain_count"].mean()
    # print(uncertain_mean)
    # absolut_mean = df0.groupby("status")["absolutist_count"].mean()
    # print(absolut_mean)
    
    
     # --- Graph: Zeit-Fokus (Oben Rechts) ---
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (16,9)) # Manuelle Achsenpositionierung
    # sns.violinplot(ax = axes[0,1],data=df0, x="time_focus_score", y="status", hue="status", legend=False)
    # axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.6) # Rote Null-Linie
    # axes[0,1].set_title("Fokus: Zukunft vs. Vergangenheit", fontsize=14)
    # axes[0,1].set_xlabel("<- Zukunft / Sorgen (0.0) Vergangenheit / Reue ->")
    # axes[0,1].set_ylabel("")

    # sns.histplot(ax = axes[0,0], data = df_plot, x = "absolutist_ratio", hue = "status", legend= True, bins = 25).set(yscale ="log")
    # sns.histplot(ax = axes[1,0], x = "uncertain_ratio", hue = "status", data = df_plot, bins = 25).set(yscale ="log")
    # #sns.pointplot(ax = axes[1,1], data = df0, x = "absolutist_ratio", y = "uncertain_ratio", hue = "status", errorbar="sd")
    # sns.scatterplot(ax = axes[1,1], data = df_plot, x = "uncertain_count", y = "absolutist_count", hue = "status").set(yscale = "log", xscale = "log")
    # #plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    # fig.savefig("pronoun_analysis_5.png")
    # plt.show()

    # sns.jointplot(
    #     data=df_plot, 
    #     y="absolute_uncertain_ratio", 
    #     x="time_focus_score", 
    #     hue="status", 
    #     kind="kde",
    #     levels = 2
    # )
    # plt.show()
    # sns.jointplot(
    #     data=df_plot, 
    #     y="absolutist_count", 
    #     x="uncertain_count", 
    #     hue="status", 
    #     kind="kde",
    #     levels = 3,
    #     ylim = (0,30),
    #     xlim = (0,30)
    # )
    # plt.show()
    # os.chdir(path)

    absolutist_words = ["always", "never", "completely", "nothing", "everything", "all",
                              "ever", "everyone", "i know",
                                "constantly", "every","full", "done","finish","end"]
    uncertain_words = ["maybe", "might", "could",
                        "i think", "not sure",
                        "weird","can't"]
    negative_words = ["die", "kill", "pain", "hurt", "cry", "sad", "alone", "lonely", "hate",
                       "bad", "tired", "worse", "hopeless", "empty", "scared","disturbed","nuts",
                       "dead","psycho","mad","insane","freak","scary","stressed","stress","embarrassed","embarrassing",
                       "frustrating","frustrated","isolated","isolation","isolating","mental","ill",
                        "brain","forced","demand","abuse","sexual","sex","life","meds","medications","medication"]
    
    # Hilfsfunktion: Zählt die Wörter einer Liste, bildet den Durchschnitt und berechnet den Z-Score
    def get_z_score_df(word_list):
        counts = {}
        for word in word_list:
            pattern = rf'(?i)\b{word}\b'
            counts[word] = df0['statement'].str.count(pattern).groupby(df0['status']).mean().to_dict()
        
        df = pd.DataFrame(counts)
        df.to_json("word_frequency.json")
        # Z-Score Standardisierung. .fillna(0) verhindert Fehler, falls ein Wort zufällig 0 mal vorkommt.
        df_z = ((df - df.mean()) / df.std()).fillna(0)
        return df_z

    # Daten für alle drei Kategorien berechnen
    df_abs_z = get_z_score_df(absolutist_words)
    df_unc_z = get_z_score_df(uncertain_words)
    df_neg_z = get_z_score_df(negative_words)


    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 9))
    fig.suptitle("Wort-Signaturen: Absolutismus, Unsicherheit und negative Emotionen", fontsize=18, fontweight='bold', y=0.98)

    # --- Heatmap Absolutismus ---
    sns.heatmap(df_abs_z, ax=axes[0], cmap="coolwarm", center=0, linewidths=.5, cbar_kws={'label': 'Z-Score (Rot = Stark)'})
    axes[0].set_title("1. Absolutistische Wörter (Alles-oder-Nichts-Denken)", fontsize=14)
    axes[0].set_ylabel("Status")
    axes[0].set_xlabel("") # Verstecken, da es sonst zu überladen wirkt

    # --- Heatmap Unsicherheit ---
    sns.heatmap(df_unc_z, ax=axes[1], cmap="coolwarm", center=0, linewidths=.5, cbar_kws={'label': 'Z-Score (Rot = Stark)'})
    axes[1].set_title("2. Unsichere Wörter (Zweifel und Abwägung)", fontsize=14)
    axes[1].set_ylabel("Status")
    axes[1].set_xlabel("")

    # --- Heatmap Negative Wörter ---
    sns.heatmap(df_neg_z, ax=axes[2], cmap="coolwarm", center=0, linewidths=.5, cbar_kws={'label': 'Z-Score (Rot = Stark)'})
    axes[2].set_title("3. Negative / Emotionale Wörter (Schmerz, Trauer, Erschöpfung)", fontsize=14)
    axes[2].set_ylabel("Status")
    axes[2].set_xlabel("Spezifische Wörter", fontsize=12)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    

    plt.savefig(os.path.join(output_path, "three_word_heatmaps.png"))
    
    plt.show()
    def get_df(word_list):
        counts = {}
        for word in word_list:
            pattern = rf'(?i)\b{word}\b'
            counts[word] = df0['statement'].str.count(pattern).groupby(df0['status']).mean().to_dict()
        
        df = pd.DataFrame(counts)
        return df
    df_abs = get_df(absolutist_words)
    print(df_abs)
    df_unc = get_z_score_df(uncertain_words)
    print(df_unc)
    df_neg = get_z_score_df(negative_words)
    print(df_neg)
    df_con =  pd.concat([df_abs,df_neg,df_unc],axis=1)
    os.chdir(path)
    return df_con



#----------------------------------------------------------------------------------------------------


def pronouns(df0:pd.DataFrame, statement_len_filter = 0):
    """
    gibt die durchschnittliche vorkommens häufigkeit von verschiedenen Pronomen an und aus, sowie eine Tabelle dazu.
    benötigt einen DataFrame als übergabe parameter.
    \nstatement_len_filter sortiert alle statements aus, deren wörter länge kleiner als der übergebene int ist. (Default = 0)
    """
    path = os.getcwd()
    output_path = os.path.join(path, "Output")
    os.chdir(output_path)
    min_statement_len = statement_len_filter #!!!!ACHTUNG aktuell werden nur statements mit dieser Anzahl genommen und alle anderen aussortiert!!!!
    
    self_pronouns = r'(?i)\b(we|us|ourself|ourselves|i|me|mine|myself|my)\b'
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
    words_over = df0['word_count'] > min_statement_len
    df0 = df0[words_over]
    # wie oft werden ich bezogene Pronomen im verhältniss zur länge der Nachricht verwendet
    df0['self_pronoun_to_word_ratio'] = df0['self_pronouns_count'] / df0['word_count']
    df0['other_pronouns_ratio'] = df0['other_pronouns_count'] / df0['word_count']
    # die Differenz der selbstbezogenen Pronomen und der anderen Pronomen 
    df0['dif_pronouns'] = df0['self_pronouns_count'] - df0['other_pronouns_count']
    

    self_pr = ["we", "us", "ourself","ourselves", "i", "me", "mine", "myself", "my", "i'll", "we'll", "i'm"]
    other_pr = ["you", "your", "yours", "yourself", "she", "her", "hers", "herself", "he", "him", "his", "himself", "they", "them", "their", "themselves","themself"]

    counter_self = {}
    counter_other = {}

    # 1. Selbst-Pronomen zählen
    for pron in self_pr:
        # Suchmuster bauen: (?i) für ignoriere Groß-/Kleinschreibung, \b für exaktes Wort
        pattern = rf'(?i)\b{pron}\b'
        # Zählen, nach Status gruppieren, summieren und ins Dictionary packen
        counter_self[pron] = df0['statement'].str.count(pattern).groupby(df0['status']).mean().to_dict()

    # 2. Andere-Pronomen zählen
    for pron in other_pr:
        pattern = rf'(?i)\b{pron}\b'
        counter_other[pron] = df0['statement'].str.count(pattern).groupby(df0['status']).mean().to_dict()
    
    # --- DAS ERGEBNIS WUNDERSCHÖN DARSTELLEN ---
    df_self_counts = pd.DataFrame(counter_self)
    df_other_counts = pd.DataFrame(counter_other)

    print("\n--- Absolute Anzahl der ICH-Pronomen pro Status ---")
    print(df_self_counts)

    print("\n--- Absolute Anzahl der ANDEREN-Pronomen pro Status ---")
    print(df_other_counts)
    print("\n--- durschnittliche Wortanzahl ---")
    print(df0.groupby("status")["word_count"].mean())
    print(df0["statement"].str.len().groupby(df0["status"],sort=True).idxmax())


    # 1. Beide Tabellen für eine Master-Übersicht zusammenfügen
    df_all_counts = pd.concat([df_self_counts, df_other_counts], axis=1)

    # 2. Spaltenweise Standardisierung (Z-Score)
    # (Wert - Durchschnitt der Spalte) / Standardabweichung der Spalte
    df_heatmap = (df_all_counts - df_all_counts.mean()) / df_all_counts.std()

    # 3. Die Heatmap zeichnen
    plt.figure(figsize=(18, 8))
    
    # cmap="coolwarm" macht tiefe Werte blau, Durchschnitt weiß und hohe Werte rot
    # center=0 zentriert die Farbskala exakt auf den Durchschnitt
    sns.heatmap(
        df_heatmap, 
        cmap="coolwarm", 
        vmax= 1.5,
        vmin = -1.5,
        annot=False,
        linewidths=.5,  
        cbar_kws={'label': 'Nutzung (Blau = Unterdurchschnittlich, Rot = Überdurchschnittlich)'}
    )

    plt.title("Pronomen-Signatur: Welche Gruppe nutzt welche Pronomen überdurchschnittlich?", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Pronomen", fontsize=12)
    plt.ylabel("Psychologischer Status", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("pronoun_heatmap_min.png")
    plt.show()
    
    sns.lmplot(
        data=df0, 
        x="word_count", 
        y="self_pronoun_to_word_ratio", 
        hue="status", 
        scatter_kws={'alpha': 0.6, 's': 10},
        line_kws={'linewidth': 3},     
        height=7, 
        aspect=1.5,
        palette="husl"
    )

    # Normalisieren der Differenz, bzw wie dominant sind die selbstbezogenen Pronomen auf einer Skala von -1 bis 1
    # words_over = df0['word_count'] > 0
    # df0 = df0[words_over]
    # mask = df0['all_pronouns_count'] > 40
    # df_plot = df0[mask]
    # sns.set_theme(style="whitegrid", palette="husl")
    # sns.jointplot(
    #     data=df_plot, 
    #     y="self_pronoun_to_word_ratio", 
    #     x="other_pronouns_ratio", 
    #     hue="status", 
    #     kind="scatter",
    #     ylim =(0,0.25),
    #     xlim = (0,0.25) #,
    #     # levels = 2
    # )
    
    # plt.show()
    os.chdir(path)
    print(df_all_counts.head())
    return df_all_counts

def depr_vs_suic(df0 : pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    df = df0[df0['status'].isin(["Suicidal", "Depression"])].reset_index(drop=True)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(df['statement'])
    feature_names = tfidf.get_feature_names_out()
    suicidal_idx = df.index[df['status'] == 'Suicidal'].tolist()
    depression_idx = df.index[df['status'] == 'Depression'].tolist()
    suicidal_means = np.asarray(X_train_tfidf[suicidal_idx].mean(axis=0)).flatten()
    depression_means = np.asarray(X_train_tfidf[depression_idx].mean(axis=0)).flatten()
    word_scores = pd.DataFrame({
        'word': feature_names,
        'suicidal_score': suicidal_means,
        'depression_score': depression_means
    })
    word_scores['diff'] = word_scores['suicidal_score'] - word_scores['depression_score']
    top_suicidal_words = word_scores.sort_values(by='diff', ascending=False).head(20)
    top_depression_words = word_scores.sort_values(by='diff', ascending=True).head(20)
    filtered = pd.concat([top_suicidal_words, top_depression_words], axis=0)
    fig = plt.figure(figsize=(16,9))
    sns.set_style("whitegrid")
    ax = sns.scatterplot(
        data=filtered, 
        x="depression_score", 
        y="suicidal_score", 
        hue="diff", 
        palette="coolwarm", 
        s=120,
        edgecolor='w',
        linewidth=1
    )
    for i in range(filtered.shape[0]):
        ax.text(
            filtered.iloc[i]['depression_score'] + 0.001, # slightly shift text to the right
            filtered.iloc[i]['suicidal_score'], 
            filtered.iloc[i]['word'], 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )
    
    # Add a diagonal reference line (where suicidal_score == depression_score)
    max_val = max(filtered['depression_score'].max(), filtered['suicidal_score'].max())
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', alpha=0.5)
    
    # Title and Labels
    ax.set_title("Distinguishing Words: Suicidal vs. Depression TF-IDF Scores", fontsize=14, pad=15)
    ax.set_xlabel("Depression Score (Mean TF-IDF)", fontsize=12)
    ax.set_ylabel("Suicidal Score (Mean TF-IDF)", fontsize=12)
    
    # Save the chart safely
    plt.savefig("suic_vs_depr.png", bbox_inches='tight')
    
    return top_suicidal_words, top_depression_words