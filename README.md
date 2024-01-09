# Entwicklung eines Chatbots für Themen rund um die Fußball Weltmeisterschaften
Ein Projekt im Modul Aktuelle Data Science Entwicklungen - Natural Language Processing (WWI20DSA).

## Idee
Für die Sportart Fußball soll ein Nutzer die Möglichkeit haben, sich auf Englisch über den Chatbot verschiedene Informationen zu den FIFA Weltmeisterschaften anzeigen lassen zu können.  

![Alt text](/static/images/demo.png)

## Mögliche Fragen an den Chatbot
- Which team became [ORDINAL] place in [YEAR]?
- On which position was [COUNTRY] in [YEAR]?
- Who hosted the World Cup in the year [YEAR]?
- On average, how many goals were scored in [YEAR]?
- How many goals were scored [YEAR]?
- How many matches were played in the world-championship of [YEAR]?

## Ausführen des Chatbots
1. Klonen des Repositories

```git clone https://github.com/bgrnaymane/NLP-Projekt.git```

2. Ausführen folgender Befehle:

```python -m spacy download en_core_web_sm```

```cd NLP-Projekt```

```pip install -r requirements.txt```

```python app.py```
- Wenn das NLTK-Package Stopwords nicht installiert wird, folgenden Code ausführen:
```
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
```

## Natural Language Processing
### Pre-Processing
Der textuelle Input des Users wird in den Vorverarbeitungsschritten in für die Maschinen verständliche Sprache umgewandelt. Dafür wird der Text in Kleinschreibung (lowercase) umformuliert, um so eine Vereinheitlichung des Texts zu erwirken. Zudem werden jegliche Satzzeichen entfernt.

Das SpaCy Modell "en_core_web_sm" wird genutzt, um die Stopwords zu entfernen. Sogenannte Stopwords bezeichnen für die inhaltliche Aussagekraft des Texts nicht aufwertende Begriffe, wie beispielsweise Artikel, Konjunktionen, usw. die dann entfernt werden, um inhaltlich wertvolleren Text zu erhalten.

### Numerizer 
SpaCy Extension, um Zahlen in englischer Sprache in Integer, bzw. Floatwerte umzuwandeln.
- one -> 1
- Umwandlung auch bei abgewandelten Formen
  - the second -> the 2nd
### Intentanalyse
***„Wie können wir mit unserem Chatbot unseren Usern im Kontext unserer Produkte und der Customer Journeys helfen?“***
- Herausfinden was der User möchte
- Schlagwörter dem eigentlichen Zweck zuordnen
- Auf Basis des Intents, weitere Aufgaben des Chatbots anstoßen
- Teil der Natural Language Understanding Domäne

**Support Vector Classifier** ist ein Algorithmus für maschinelles Lernen, der für Klassifizierungsaufgaben verwendet wird. Er findet die beste Grenze (Hyperebene), die die Daten in verschiedene Klassen trennt, indem er die Marge (den Abstand zwischen der Grenze und den nächstgelegenen Datenpunkten aus jeder Klasse) maximiert. SVCs können sowohl für lineare als auch für nichtlineare Klassifizierungsaufgaben eingesetzt werden, indem verschiedene Kernel-Funktionen wie lineare, polynomiale und radiale Basisfunktionen (RBF) verwendet werden. Sie gelten als leistungsfähige Klassifikatoren mit guter Genauigkeit, die besonders nützlich sind, wenn die Daten nicht linear trennbar sind und die Anzahl der Merkmale viel größer ist als die Anzahl der Stichproben.

**Word2Vec** ist eine Technik zur Erstellung von Worteinbettungen, d.h. numerischen Darstellungen von Wörtern. Die Idee hinter den Worteinbettungen ist es, Wörter in einem hochdimensionalen Raum so darzustellen, dass semantisch ähnliche Wörter nahe beieinander liegen und unähnliche Wörter weit voneinander entfernt sind. Word2Vec verwendet ein neuronales Netz, um diese Repräsentationen aus einem großen Korpus von Textdaten zu lernen. Die daraus resultierenden Worteinbettungen können für eine Vielzahl von Aufgaben der Verarbeitung natürlicher Sprache verwendet werden.

**TF-IDF** ist eine Technik zur Darstellung der Bedeutung von Wörtern in einem Dokument. Die Idee hinter TF-IDF ist es, Wörter danach zu gewichten, wie häufig sie in einem Dokument vorkommen, wobei auch berücksichtigt wird, wie häufig sie in einem Korpus von Dokumenten vorkommen. Der TF-IDF-Wert eines Wortes ist das Produkt aus seiner Termfrequenz (TF) und seiner inversen Dokumentenfrequenz (IDF). Wörter, die nur in einem bestimmten Dokument vorkommen, haben einen höheren TF-IDF-Wert als Wörter, die in vielen Dokumenten vorkommen.

**Gridsearch** ist eine Funktion, die dabei hilft, die beste Kombination von Hyperparametern für ein bestimmtes Modell des maschinellen Lernens zu finden. Dazu wird das Modell mit allen möglichen Kombinationen von Hyperparameterwerten trainiert und die Kombination zurückgegeben, die die beste Punktzahl gemäß der angegebenen Scoring-Metrik ergibt.

**k-fold** ist eine Technik zur Bewertung der Leistung eines maschinellen Lernmodells, bei der die Daten in k Teilmengen aufgeteilt werden, das Modell auf k-1 der Teilmengen trainiert und auf der verbleibenden Teilmenge bewertet wird. Der Prozess wird k-mal wiederholt und die Leistung wird über alle k Iterationen gemittelt.

### Named Entity Recognition
Erkennen einzelner Entitäten, um auf Basis eines erkannten Intents die richtigen Daten aus den CSV-Files einzulesen.
Folgende Entitäten wurden dazu genutzt und mithilfe von SpaCy erkannt:
- DATE: Erkennen der Jahreszahl, in der eine Weltmeisterschaft stattgefunden hat
- ORDINAL: Erkennen der Platzierung eines Landes bei einer Weltmeisterschaft
  - Zur Erkennung mussten teilweise die Wörter zunächst mithilfe der SpaCy Extension Numerizer umgewandelt und die Zahlenwerte daraus extrahiert werden (first -> 1st -> 1)
 - GPE (Geopolitical Entity): Erkennung eines bestimmten Landes in dem vorliegenden Satz

## Chatbot
- ~Open-Source Rasa Chatbot (Fragen ob Alternative verwendet werden soll)~ -> eigenes Modell entwickelt und trainiert
- ~APIs zum Anbinden/Integrieren der relevanten Daten~(Nicht umsetzbar, da APIs kosten) -> CSV-Files zu FIFA Weltmeisterschaften als Datenbasis 
- Backend mit Python-Flask
- Frontend mit HTML (CSS) inklusive Javascript Funktion (zur Umsetzung des Chatbot)


### Backend mit Python-Flask
- Anbindung der Data Processing-Funktionen, um Intents zu erkennen und die Daten aus den CSV-Files zu ziehen
- Datentransfer mithilfe von JSON-Files, um die Kommunikation zwischen dem Javascript Chatbot und den Pythonfunktionen zu ermöglichen

### Frontend (HTML, CSS, Javascript)
Vorlage: https://github.com/patrickloeber/chatbot-deployment
- Zulassen von Userinput durch den Chatbot und weitergeben dieses Inputs an die Pythonfunktionen
- Zurückgeben der Antwort des Chatbots und Darstellen in dem zugehörigen Fenster

### Funktionen des Chatbots
- Visualisierung der Ausgabe
- Interaktion: Kunde/Anwender kann dem Chatbot Fragen stellen, welche vom Chatbot beantwortet werden
- Named Entity Recognition: um Länder(Geopolitical Entity), Ordinalwerte(Platzierungen) und Date erkennen und ausgeben lassen zu können
  - Ersetzen von vordefinierten Synonymen (bsp. Weltmeister = 1. Place)
- Numerizer: Erkennung von geschriebenen Zahlen (bsp. one) 
- Chatbot akzeptiert lediglich die englische Sprache
- Intent Detection: Die Intention hinter der Anfrage des Nutzer verstehen und eine Abfrage darauf aufbauend machen 

### Quellen
**Intent Detection**
- Liu, Bing, and Ian Lane. "Attention-based recurrent neural network models for joint intent detection and slot filling." arXiv preprint arXiv:1609.01454 (2016).
- https://www.atlantis-press.com/journals/hcis/125963694
- Cahn, Jack. "CHATBOT: Architecture, design, & development." University of Pennsylvania School of Engineering and Applied Science Department of Computer and Information Science (2017).
- https://www.kaggle.com/code/taranjeet03/intent-detection-svc-using-word2vec/notebook#)

**Spacy**
- Numerizer: https://github.com/jaidevd/numerizer
- https://spacy.io/usage/linguistic-features
- Model "en_core_web_trf": https://spacy.io/usage/models

**Datengrundlage**
- https://www.kaggle.com/datasets/iamsouravbanerjee/fifa-football-world-cup-dataset

**Website Entwicklung**
- https://github.com/patrickloeber/chatbot-deployment

**Bilder**
- https://wallpapersden.com/2022-fifa-world-cup-hd-wallpaper/


