import os

#Finde den Pfad zu dem Datasets Ordner:
path = os.getcwd()
output_path = os.path.join(path, "../output")
data_path = os.path.join(path, "../Datasets")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Explorative Daten analyse:
def expl_data():
    """Verarbeitet die Data und returned einen Dataframe"""
    from src.EDA import explorativ_data_analysis_copy as eda
    #import data_loading as dl
    import datenbereinigung as dc
    #import feature_corralation_test as fct
    #import feature_builder as fb
    #import Klassifikation_Modell_Training as KMT
    df0 = dc.clean_data()

    #fct.correlation_test(eda.expl_data(df0))
    #df0, all_target_words = fb.create_all_features(df0)
    #KMT.train_testdaten_split(df0)
    #eda.pronouns(df0)
    print(eda.depr_vs_suic(df0))

expl_data()