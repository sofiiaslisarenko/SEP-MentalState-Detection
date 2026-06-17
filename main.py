import pandas as pd                 #Für das Erstellen und Arbeiten mit Dataframes
import os                           #Zum finden des Paths (File Ordner)
import matplotlib.pyplot as plt     #Zusammen mit seaborn zum erstellen von Graphen
import re                           #Für Regex Operationen
import seaborn as sns

#Finde den Pfad zu dem Datasets Ordner:
path = os.path.join(os.getcwd(), "Datasets")
print(f"Path to datasets: {path}")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Erstellen einer Liste mit den csv Dateien und erste Auswertungen
file_get = [file for file in os.listdir(path)]
print(file_get)

df0 = pd.read_csv(file_get[0])
df1 = pd.read_csv(file_get[1])

#print(df0.head())       #df0 enthält Combined Data.csv
#print(df1.head())       #df1 enthält Suicide_detection.csv

# cleanup und merging der beiden Dataframes, falls kompatibel

# Unique Werte in den Dataframes überprüfen, bzw Welche Werte sich Doppeln
# Und ob das für die Klassifikation relevant sein könnte, oder ob es sich um Rauschen handelt, welches entfernt werden sollte
#print(df0.nunique())
#print(df1.nunique())

# Klassifikation zum finden, oder testen von Features, welche relevant sein könnten

# Sebastians Explorative Daten analyse:
def expl_data():
    """Graphische Darstellung der Daten zum Visuellen analysieren."""
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
    #normalisieren von ergebnissen:
    
    #Bauen des Graphen mit seaborn (sns)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.barplot(ax = axes[0], x=anteil_self_to_all.index, y=anteil_self_to_all.values, hue=anteil_self_to_all.index, legend=True)
    axes[0].set_ylabel("Ratio of self pronouns to all pronouns in %")
    axes[0].set_yticklabels([f'{x*100:.0f}%' for x in axes[0].get_yticks()])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    sns.boxenplot(ax = axes[1], x="self_pronoun_to_word_ratio", y="status", data=df0, hue = "status")
    plt.show()
    fig, axes = plt.subplots(nrows=2, ncols=3)
    sns.histplot(ax = axes[0,0],data=df0, x="self_pronoun_to_word_ratio", hue="status", kde=True, bins = 40)
    sns.histplot(ax = axes[0,1],data=df0, x="self_pronouns_count", hue="status", kde=False, log_scale=False, bins = 30, binrange = (5,100))
    sns.histplot(ax = axes[1,1],data=df0, x="all_pronouns_count", hue="status", kde=False, log_scale=False, bins = 30, binrange = (5,100))
    sns.histplot(ax = axes[1,2],data=df0, x="all_pronouns_count", hue="status", kde=False, log_scale=False, bins = 30, binrange = (0,6))
    sns.histplot(ax = axes[1,0],data=df0, x="word_count", hue="status", kde=True,log_scale=True)
    sns.violinplot(ax = axes[0,2],data=df0, y = "pronoun_dominance", x = "status", hue = "status")

    plt.show()


    # Absolutistische Wörter (Alles oder Nichts Denken)
    absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all)\b'
    
    # Zeitliche Orientierung (Past und Future)
    # Erstmal wenige standart wörter
    past_words = r'(?i)\b(was|were|had|did|been|could)\b'
    future_words = r'(?i)\b(will|shall|going to|might|worry|worried|anxious)\b'
    
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
expl_data()