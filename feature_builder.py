import pandas as pd
import re

def create_all_features(df0: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Makro-Metriken und Token-Frequenzen für jedes einzelne Ziel-Wort."""
    
    # 1. Wortanzahl (Verhindert Division durch 0, indem 0 zu 1 wird)
    df0['word_count'] = df0["statement"].str.split().str.len().replace(0, 1)
    
    # === MAKRO-FEATURES ===
    self_pronouns = r'(?i)\b(we|us|ourself|ourselves|i|me|mine|myself|my|i\'ll|we\'ll|i\'m)\b'
    other_pronouns = r'(?i)\b(you|your|yours|yourself|yourselves|he|she|her|hers|herself|him|his|himself|they|them|their|theirs|themselves|thyself|thine)\b'
    # Absolutistische Wörter
    absolutist_words = r'(?i)\b(always|never|absolutely|completely|nothing|everything|entirely|all|nobody|forever|ever|noone|everyone|everybody|i know|impossible|must)\b'
    uncertain_words = r'(?i)\b(maby|perhaps|possibly|possible|may|might|could|i think|not sure|uncertain)\b'
    # Zeitliche Orientierung
    past_words = r'(?i)\b(was|were|had|did|been|could|said|went|ago|last|got|wanted|used|liked)\b'
    future_words = r"(?i)\b(will|shall|going to|might|worry|worried|anxious|'ll)\b"
    
    df0['self_pronouns_count'] = df0["statement"].str.count(self_pronouns)
    df0['other_pronouns_count'] = df0["statement"].str.count(other_pronouns)
    df0['all_pronouns_count'] = df0['self_pronouns_count'] + df0['other_pronouns_count']
    df0['pronoun_dominance'] = ((df0['self_pronouns_count'] - df0['other_pronouns_count']) / df0['all_pronouns_count']).fillna(0)
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
    
    self_pr_list = ["we", "us", "ourself", "i", "me", "mine", "myself", "my", "i'll", "we'll", "i'm"]
    other_pr_list = ["you", "your", "yours", "yourself", "yourselves", "he", "she", "her", "hers", "herself", "him", "his", "himself", "they", "them", "their", "their's", "themselves"]
    
  
    absolutist_words = ["always", "never", "completely", "nothing", "everything", "all", "ever", "everyone", "i know", "constantly", "every", "full", "done", "finish", "end"]
    uncertain_words = ["maybe", "might", "could", "i think", "not sure", "weird", "can't"]
    negative_words = ["die", "kill", "pain", "hurt", "cry", "sad", "alone", "lonely", "hate", "bad", "tired", "worse", "hopeless", "empty", "scared", "disturbed", "nuts", "dead", "psycho", "mad", "insane", "freak", "scary", "stress", "embarrassed", "embarrassing", "frustrating", "frustrated", "isolated", "isolation", "isolating", "mental", "ill", "brain", "forced", "demand", "abuse", "sexual", "sex", "life"]

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
    
    return df0.fillna(0), list(seen_features)