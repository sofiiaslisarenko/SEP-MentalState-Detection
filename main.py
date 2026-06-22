import pandas as pd
import os

#Finde den Pfad zu dem Datasets Ordner:
path = os.getcwd()
output_path = os.path.join(path, "Output")
data_path = os.path.join(path, "Datasets")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Explorative Daten analyse:
def expl_data_seb():
    """Verarbeitet die Data und returned einen Dataframe"""
    import explorativ_data_analysis_copy as eda
    #import data_loading as dl
    import data_clean as dc
    #import feature_corralation_test as fct
    df0 = dc.clean_data()

    #fct.correlation_test(eda.expl_data(df0))
    eda.absolute_uncertain(df0)
    #eda.pronouns(df0)

expl_data_seb()