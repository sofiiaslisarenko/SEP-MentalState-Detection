# SEP-Projekt-Depressionen


Dataset used:

https://www.kaggle.com/datasets/szegeelim/mental-health/data

How to setup:
Für Windows:
1. workspace erstellen

2. repositorie clonen
    git clone https://github.com/sofiiaslisarenko/SEP-Projekt-Deppressionen

3. envoirement erstellen:
    (Bei PyCharm direkt beim Workspace erstellen)
    -   Das envoirement dient dazu, das Module per pip nicht "in alle Projekte geladen" werden und so Probleme verursachen.
        Beispielsweise können Versionskonflikte auftreten (PyTorch und NumPy sind bekannt dafür).

    einfach die setup.bat datai ausführen, dann im CMD Terminal:
        venv\Scripts\activate

    alternativ manuel im cmd Terminal:
        python -m venv <your_enovirement_name>   (statt your_enovirement_name einen beliebigen namen wählen)
        <your_enovirement_name>\Scripts\activate
        pip install -r requirements.txt

> **Hinweis:** `src/` muss als Sources Root in PyCharm eingestellt werden (Rechtsklick auf `src/` → *Mark Directory as* → *Sources Root*).



Allgemeine Informationen


Dataset

Der Datensatz Combined Data (https://www.kaggle.com/datasets/szegeelim/mental-health/data) ist wurde von uns auf Kaggle gefunden und enthält
gelabelten Text von Socialmediaplatformen. Wie dem Namen zu entnehmen ist, wurde dieser aus anderen Kaggle Datensätzen zusammengesetzt.
Dem Text ist jeweils eins von 7 Labels zugeornet ('status'): Anxiety, Stress, Bipolar, Depression, Normal, Personality disorder, Suicidal.
Die Label sind keine medizinischen Diagnosen, sondern reflektieren lediglich den Kontext indem die Textbeiträge gefunden wurden z.B. bestimmte subreddits.


Unsere Fragestellungen

