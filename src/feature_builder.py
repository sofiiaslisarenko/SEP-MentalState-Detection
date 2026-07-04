import pandas as pd


def create_all_features(df0: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Makro-Metriken und Token-Frequenzen für jedes einzelne Ziel-Wort um sie als einen gesammelten DataFrame weiter zu geben."""
    
    # 1. Wortanzahl (Verhindert Division durch 0, indem 0 zu 1 wird)
    df0['word_count'] = df0["statement"].str.split().str.len().replace(0, 1)
    
    # === MAKRO-FEATURES ===
    self_pronouns = r'(?i)\b(i|me|mine|myself|my|i\'ll|i\'m)\b'
    first_pl_pr = r"(?i)\b(us|we|we'll|our)\b"
    second_pronouns = r'(?i)\b(you|your|yours|yourself|yourselves)\b'
    third_pr = r'(?i)\b(he|she|her|hers|him|his)\b'
    other_pl_pr = r"(?i)\b(they|them|their|theirs|themselves|themself)\b"
    self_pr_other = r"(?i)\b(himself|herself|themselves|themself)\b"
    # Absolutistische Wörter
    absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all|nobody|forever|ever|noone|everyone|everybody|i know|impossible|must)\b'
    uncertain_words = r'(?i)\b(maby|perhaps|possibly|possible|may|might|could|i think|not sure|uncertain)\b'
    # Zeitliche Orientierung
    past_words = r"(?i)\b(was|were|had|did|been|could|said|went|ago|last|got|wanted|used|liked|have been|'ve|used to|past)\b"
    future_words = r"(?i)\b(will|shall|going to|might|worry|worried|anxious|'ll|'d like to|might|should|future)\b"
    
    # Schlaf-bezogene Wörter (Feature von Sofiia)
    df0['sleep_words'] = df0['statement'].str.lower().str.count(
        r'\bsleep\b|\binsomnia\b|\bnight\b|\btired\b|\bawake\b|\basleep\b|\bsleepless\b|\bsleeping\b|\bwake\b'
    )

    df0['self_pronouns_count'] = df0["statement"].str.count(self_pronouns)
    df0['first_pl_pr_count'] = df0["statement"].str.count(first_pl_pr)
    df0['second_pronouns_count'] = df0["statement"].str.count(second_pronouns)
    df0['third_pr_count'] = df0["statement"].str.count(third_pr)
    df0['other_pl_pr_count'] = df0["statement"].str.count(other_pl_pr)
    df0['self_pr_other_count'] = df0["statement"].str.count(self_pr_other)
    df0['all_pronouns_count'] = df0['self_pronouns_count'] + df0['other_pronouns_count']
    df0['absolutist_count'] = df0["statement"].str.count(absolutist_words)
    df0['uncertain_count'] = df0["statement"].str.count(uncertain_words)
    df0['past_count'] = df0["statement"].str.count(past_words)
    df0['future_count'] = df0["statement"].str.count(future_words)
    df0['total_time_words'] = df0['past_count'] + df0['future_count']

    
    # Zeichensetzung als Emotions-Barometer (Fragezeichen und Resignation/Pausen)
    # Der Backslash \ ist wichtig, weil ? und . in Regex Sonderzeichen sind.
    df0['question_marks_count'] = df0["statement"].str.count(r'\?')
    df0['ellipses_count'] = df0["statement"].str.count(r'\.\.\.')
    df0['exclamation_marks_count'] = df0["statement"].str.count(r'\!')
    print(df0.head())
    df0.info()
    self_pr_list = ["we", "us", "ourself", "i", "me", "mine", "myself", "my", "i'll", "we'll", "i'm","our"]
    other_pr_list = ["you", "your", "yours", "yourself", "yourselves", "he", "she", "her", "hers", "herself", "him", "his", "himself", "they", "them", "their", "their's", "themselves", "themself"]
    
  
    absolutist_words = ["always","anymore","want", "never", "completely", "nothing", "everything", "all", "ever", "everyone", "i know", "constantly", "every", "full", "done", "finish", "end"]
    uncertain_words = ["maybe","don't", "insufficient","might", "could", "i think", "not sure", "weird", "can't", "feel"]
    negative_words = ["depression","work","school","fucking","die", "kill", "suicide","pain", "hurt", "cry", "sad", "alone", "lonely", "hate", "bad", "tired", "worse", "hopeless", "empty", "scared", "disturbed", "nuts", "dead", "psycho", "mad", "insane", "freak", "scary", "stress", "embarrassed", "embarrassing", "frustrating", "frustrated", "isolated", "isolation", "isolating", "mental", "ill", "brain", "forced", "demand", "abuse", "sexual", "sex", "life","meds","medications","medication"]

    all_target_words = self_pr_list + other_pr_list + absolutist_words + uncertain_words + negative_words

    new_features = {}
    word_count = df0["statement"].str.split().str.len().replace(0, 1)
    
    seen_features = set()
    for word in all_target_words:
        clean_name = f"freq_{word.replace(' ', '_').replace(chr(39), '')}"
        if clean_name not in seen_features:
            pattern = rf'(?i)\b{word}\b'
            # Hier nur in das Dict schreiben
            new_features[clean_name] = df0["statement"].str.count(pattern) / word_count
            seen_features.add(clean_name)
            
    # 2. EINMALIGES JOINEN (Das löst die Fragmentierung)
    df_new_features = pd.DataFrame(new_features)
    df0 = pd.concat([df0, df_new_features], axis=1)
    df0.info()
    return df0.fillna(0), list(seen_features)



# Features von Markus
import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords', quiet=True)
stw_eng = set(stopwords.words('english'))
satzzeichentabelle = str.maketrans('', '', string.punctuation)

def text_process(mess):
    text_ohne_sz = mess.translate(satzzeichentabelle)
    clea = []
    for wort in text_ohne_sz.split():
        wort_lower = wort.lower()
        if wort_lower not in stw_eng:
            clea.append(wort_lower)
    return clea

def text_process_with_stw(mess):
    text_ohne_sz = mess.translate(satzzeichentabelle)
    clea = []
    for wort in text_ohne_sz.split():
        clea.append(wort.lower())
    return clea

def create_additional_features(df0: pd.DataFrame) -> pd.DataFrame:
    df0['statement'] = df0['statement'].fillna("").astype(str)
    df0['tokenized_text'] = df0['statement'].apply(text_process)
    df0['tokenized_text_with_stw'] = df0['statement'].apply(text_process_with_stw)
    anz_ohne_stw = df0['tokenized_text'].str.len()
    anz_mit_stw = df0['tokenized_text_with_stw'].str.len()
    df0['stopwords_per_text_ratio'] = (anz_mit_stw - anz_ohne_stw) / anz_mit_stw.replace({0: 1})

    return df0