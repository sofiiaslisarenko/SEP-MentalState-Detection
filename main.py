import pandas as pd
import os

#Finde den Pfad zu dem Datasets Ordner:
path = os.getcwd()
output_path = os.path.join(path, "Output")
data_path = os.path.join(path, "Datasets")

#Ändere das Arbeitsverzeichniss in den gefundenen Pfad:
os.chdir(path)

# Sebastians Explorative Daten analyse:
def expl_data_seb():
    import explorativ_data_analysis as eda
    import data_loading as dl
    import feature_corralation_test as fct
    df0 = dl.load_data()
    #fct.correlation_test(eda.expl_data(df0))
    import test_run
    test_run.test_run(eda.expl_data(df0))
expl_data_seb()