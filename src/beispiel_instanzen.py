import matplotlib.pyplot as plt
import pandas as pd
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


def analyse_regressionsbeispiele(model_name, y_true, y_pred, df_original):
    # 1. Temporäres DataFrame für das Modell erstellen
    temp_df = df_original.copy()
    temp_df["actual"] = y_true
    temp_df["pred"] = y_pred
    temp_df["abs_error"] = (temp_df["actual"] - temp_df["pred"]).abs()

    # 2. Fälle auswählen
    ex_good = temp_df.nsmallest(1, "abs_error")
    ex_med = temp_df.iloc[(temp_df["abs_error"] - temp_df["abs_error"].median()).abs().argsort()[:1]]
    ex_bad = temp_df.nlargest(1, "abs_error")

    # 3. Zusammenführen
    examples = pd.concat([ex_good, ex_med, ex_bad])
    examples["example_type"] = ["Sehr gut", "Mittelmäßig", "Sehr schlecht"]
    examples["model"] = model_name
    return examples


# Analyse für alle 3 Modelle ausführen
res_ridge = analyse_regressionsbeispiele("Ridge", regression.y_test, regression.y_pred_ridge, regression.test_df)
res_rf = analyse_regressionsbeispiele("Random Forest", regression.y_test, regression.y_pred_rf, regression.test_df)
res_xgb = analyse_regressionsbeispiele("XGBoost", regression.y_test, regression.y_pred_xgb, regression.test_df)

# Alle Ergebnisse zusammenführen
all_reg_examples = pd.concat([res_ridge, res_rf, res_xgb])

# Ausgabe der Beispiele (nach Modell gruppiert)
print("\nBeispielhafte Instanzen pro Modell:")
print(all_reg_examples[["model", "example_type", "statement", "abs_error"]].to_string(index=False))

# Visualisierung: Vergleich der Fehler über alle Modelle
plt.figure(figsize=(12, 6))
sns.barplot(data=all_reg_examples, x="example_type", y="abs_error", hue="model", palette="viridis")
plt.title("Fehlergröße pro Modell und Beispiel-Typ")
plt.ylabel("Absoluter Fehler")
plt.xlabel("Qualität der Vorhersage")
plt.show()







