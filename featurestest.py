import pandas as pd
import matplotlib.pyplot as plt

df0 = pd.read_csv("Datasets/Combined Data.csv")
df0 = df0[['statement', 'status']].rename(columns={'statement': 'text', 'status': 'label'})
df0 = df0.dropna()

# Feature1 berechnen
# Großbuchstaben in jedem Text finden und zählen
#.str.findall(r'[A-Z]') — ищет в каждом тексте все заглавные буквы (от A до Z) и возвращает их как список.
#.groupby('label') — группирует строки по значению в колонке label (то есть отдельно собирает все Depression, отдельно все Normal и т.д.)
df0['upper_letters'] = df0['text'].str.findall(r'[A-Z]').str.len()
df0['caps_ratio'] = df0['upper_letters'] / df0['text'].str.len()

# Mittelwert pro Klasse
# Durchschnittlichen caps_ratio Wert pro Klasse berechnen und absteigend sortieren
#.sort_values(ascending=False) — сортируем результат от большего к меньшему (ascending=False = не по возрастанию, то есть по убыванию)
means = df0.groupby('label')['caps_ratio'].mean().sort_values(ascending=False)
print(means)

# Visualisierung
means.plot(kind='bar', figsize=(8, 5), color='#D85A30')
plt.title('Durchschnittlicher Anteil an Großbuchstaben pro Klasse')
plt.ylabel('Anteil Großbuchstaben')
plt.xlabel('Klasse')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_caps_ratio.png')
plt.show()




# Feature2 berechnen
#.str.lower() — переводим весь текст в нижний регистр (чтобы "I" и "i" считались одинаково)
#.str.count(r'\bi\b|\bme\b|\bmy\b') — считаем сколько раз встретилось любое из этих слов:
#\b — означает "граница слова" (чтобы не находить "i" внутри слова типа "big")
#i, me, my — сами слова которые ищем
#| — означает "или" (любое из трёх слов подходит)
df0['i_count'] = df0['text'].str.lower().str.count(r'\bi\b|\bme\b|\bmy\b')

# Mittelwert pro Klasse
means = df0.groupby('label')['i_count'].mean().sort_values(ascending=False)
print(means)

# Visualisierung
means.plot(kind='bar', figsize=(8, 5), color='#7F77DD')
plt.title('Durchschnittliche Häufigkeit von "I/me/my" pro Klasse')
plt.ylabel('Durchschnittliche Anzahl')
plt.xlabel('Klasse')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_i_count.png')
plt.show()



# Feature3 berechnen
df0['sleep_words'] = df0['text'].str.lower().str.count(
    r'\bsleep\b|\binsomnia\b|\bnight\b|\btired\b|\bawake\b'
)

# Mittelwert pro Klasse
means = df0.groupby('label')['sleep_words'].mean().sort_values(ascending=False)
print(means)

# Visualisierung
means.plot(kind='bar', figsize=(8, 5), color='#1D9E75')
plt.title('Durchschnittliche Häufigkeit von Schlafwörtern pro Klasse')
plt.ylabel('Durchschnittliche Anzahl')
plt.xlabel('Klasse')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_sleep_words.png')
plt.show()


# Feature4 berechnen
df0['absolute_words'] = df0['text'].str.lower().str.count(
    r'\balways\b|\bnever\b|\beveryone\b|\bnobody\b|\beverything\b|\bnothing\b'
)

# Mittelwert pro Klasse
means = df0.groupby('label')['absolute_words'].mean().sort_values(ascending=False)
print(means)

# Visualisierung
means.plot(kind='bar', figsize=(8, 5), color='#993556')
plt.title('Durchschnittliche Häufigkeit absoluter Wörter pro Klasse')
plt.ylabel('Durchschnittliche Anzahl')
plt.xlabel('Klasse')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_absolute_words.png')
plt.show()