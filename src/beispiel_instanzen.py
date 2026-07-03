import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import klassifikation
import regression

# 3 BEISPIELHAFTE INSTANZEN FÜR DIE KLASSIFIKATION

results_df = klassifikation.test_df.copy()

results_df["actual_status"] = klassifikation.y_test.values
results_df["pred_lr"] = klassifikation.y_pred_lr
results_df["pred_rf"] = klassifikation.y_pred_rf

proba_lr = klassifikation.model_lr.predict_proba(klassifikation.X_test_combined)
proba_rf = klassifikation.model_rf.predict_proba(klassifikation.X_test_combined)

results_df["confidence_lr"] = proba_lr.max(axis=1)
results_df["confidence_rf"] = proba_rf.max(axis=1)

results_df["correct_lr"] = results_df["actual_status"] == results_df["pred_lr"]
results_df["correct_rf"] = results_df["actual_status"] == results_df["pred_rf"]

results_df["statement_short"] = results_df["statement"].str.slice(0, 200)

example_1 = results_df[results_df["correct_lr"]].sort_values(
    "confidence_lr", ascending=False
).head(1)

example_2 = results_df[~results_df["correct_lr"]].head(1)

example_3 = results_df.sort_values(
    "confidence_lr", ascending=True
).head(1)

classification_examples = pd.concat([
    example_1,
    example_2,
    example_3
]).drop_duplicates().head(3)

classification_examples["example_type"] = [
    "Korrekt mit hoher Sicherheit",
    "Falsche Vorhersage",
    "Niedrige Sicherheit"
][:len(classification_examples)]

print("\n3 beispielhafte Instanzen für die Klassifikation:")
print(classification_examples[
    [
        "example_type",
        "statement_short",
        "actual_status",
        "pred_lr",
        "confidence_lr",
        "correct_lr",
        "pred_rf",
        "confidence_rf",
        "correct_rf"
    ]
].to_string(index=False))


plt.figure(figsize=(10, 5))

plot_data = classification_examples[
    ["example_type", "confidence_lr", "confidence_rf"]
].melt(
    id_vars="example_type",
    value_vars=["confidence_lr", "confidence_rf"],
    var_name="model",
    value_name="confidence"
)

plot_data["model"] = plot_data["model"].replace({
    "confidence_lr": "Logistic Regression",
    "confidence_rf": "Random Forest"
})

sns.barplot(
    data=plot_data,
    x="example_type",
    y="confidence",
    hue="model"
)

plt.ylim(0, 1)
plt.title("Konfidenz der Modelle für drei Klassifikationsbeispiele")
plt.xlabel("")
plt.ylabel("Konfidenz")
plt.tight_layout()
plt.savefig("../output/klassifikation_model_compar/classification_examples_lr_rf.png")
plt.show()


# 3 BEISPIELHAFTE INSTANZEN FÜR DIE REGRESSION
# Auswahl: gute Vorhersage, mittlere Abweichung, große Abweichung


# 1. Daten zusammenführen
reg_results_df = regression.test_df.copy()
reg_results_df["actual_sentiment"] = regression.y_test.values
reg_results_df["pred_xgb"] = regression.y_pred_xgb # Annahme: XGBoost ist Ihr Hauptmodell

# 2. Den absoluten Fehler berechnen (Distanz zwischen Vorhersage und Wahrheit)
reg_results_df["absolute_error"] = (reg_results_df["actual_sentiment"] - reg_results_df["pred_xgb"]).abs()

# 3. Auswahl der 3 Fälle basierend auf dem Fehler
# Fall 1: Sehr gute Vorhersage (kleinster Fehler)
ex_good = reg_results_df.nsmallest(1, "absolute_error")

# Fall 2: Mittlere Abweichung (Median-Fehler)
ex_med = reg_results_df.iloc[(reg_results_df["absolute_error"] - reg_results_df["absolute_error"].median()).abs().argsort()[:1]]

# Fall 3: Große Abweichung (größter Fehler)
ex_bad = reg_results_df.nlargest(1, "absolute_error")

# 4. Zusammenfügen
reg_examples = pd.concat([ex_good, ex_med, ex_bad])
reg_examples["example_type"] = ["Sehr gut", "Mittelmäßig", "Sehr schlecht"]

# 5. Ausgabe
print("\n3 beispielhafte Instanzen für die Regression:")
print(reg_examples[["example_type", "statement", "actual_sentiment", "pred_xgb", "absolute_error"]].to_string(index=False))

# Optional: Kurzer Vergleich der Fehler
plt.figure(figsize=(8, 4))
sns.barplot(data=reg_examples, x="example_type", y="absolute_error", color="red")
plt.title("Fehlergröße für die ausgewählten Beispiele")
plt.ylabel("Absoluter Fehler (MAE)")
plt.show()







