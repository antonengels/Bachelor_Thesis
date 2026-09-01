# Bachelorarbeit: Neuronale Modellierung realer Fahrstrategien im Schienenverkehr

Dieses Repository enthaelt den vollstaendigen technischen und schriftlichen
Arbeitsstand der Bachelorarbeit **„Neuronale Modellierung realer Fahrstrategien im
Schienenverkehr auf Basis historischer Betriebsdaten“**. Untersucht wird die
Vorhersage des ATO-Stellsignals aus historischen ETCS/ATO-Fahrdaten.

Das Projekt gliedert sich in drei aufeinander aufbauende Teile:

1. Rohmitschnitte werden zu bereinigten und standardisierten Zeitreihen-Datensaetzen
   aufbereitet.
2. Auf diesen Daten werden neuronale Modelle verglichen, trainiert und evaluiert.
3. Die Ergebnisse und die methodische Einordnung werden als LaTeX-Dokument erstellt.

## Projektstruktur

```text
02_Code/
├── Data_prep/    # Extraktion, Datenanalyse und Erzeugung der Modell-Splits
├── Model/        # Modellvergleiche, Training, Evaluation und Ergebnisplots
└── writing/      # LaTeX-Quelltext der schriftlichen Ausarbeitung
```

## Arbeitsablauf

Die Komponenten werden in folgender Reihenfolge verwendet:

```text
PCAPNG-Netzwerkmitschnitte
        |
        v
Data_prep: Feature-Extraktion, EDA und Train/Val/Test-Split
        |
        v
Model: Modellvergleich, finales Training und Testset-Evaluation
        |
        v
writing: Dokumentation von Methode, Ergebnissen und Fazit
```

`Data_prep` schreibt die finalen Parquet-Splits sowie Skalierungs- und
Feature-Metadaten nach `Model/data/`. Das Modelltraining und die Evaluation
speichern Berichte, Kennzahlen, Abbildungen und Checkpoints unter `Model/reports/`.
Ausgewaehlte Resultate werden in die schriftliche Ausarbeitung uebernommen.

## Teilprojekte

### Datenaufbereitung

[Data_prep/README.md](Data_prep/README.md) beschreibt die Verarbeitung von
PCAPNG-Mitschnitten mit `tshark`: Feature-Extraktion, Trip-Bildung,
explorative Datenanalyse und Erstellung der zeitbasierten Train/Val/Test-Splits.
Dort befinden sich auch die Python-Voraussetzungen, der vollstaendige
End-to-End-Ablauf und alle Skriptoptionen.

### Modellierung und Evaluation

[Model/README.md](Model/README.md) dokumentiert die Deep-Learning-Pipeline:
Modell- und Hyperparametervergleiche, finales LSTM-Training, Testset-Evaluation
sowie beschreibende Datenplots. Die Anleitung nennt die erwarteten Eingabedaten,
die Ausfuehrungsreihenfolge und die erzeugten Reports.

### Schriftliche Ausarbeitung

[writing/README.md](writing/README.md) enthaelt die Anleitung zum Bauen des
LaTeX-Dokuments, die Verzeichnisstruktur sowie Hinweise zur Pflege von Kapiteln,
Literatur, Abkuerzungen und Abbildungen.

## Voraussetzungen auf oberster Ebene

Die Unterprojekte werden getrennt eingerichtet und besitzen jeweils eigene
`requirements.txt`-Dateien beziehungsweise LaTeX-Abhaengigkeiten. Fuer einen
vollstaendigen Durchlauf werden benoetigt:

- Windows und Python-Umgebungen gemaess den READMEs von `Data_prep` und `Model`
- Wireshark beziehungsweise `tshark` fuer die Datenextraktion
- Eine LaTeX-Distribution mit `latexmk`, `biber` und `makeglossaries` fuer die
  schriftliche Ausarbeitung

## Einstieg

Fuer die Reproduktion der technischen Pipeline zuerst die Anleitung in
[Data_prep/README.md](Data_prep/README.md) befolgen. Sobald die Splits unter
`Model/data/` vorliegen, mit [Model/README.md](Model/README.md) fortfahren.
Die schriftliche Arbeit wird unabhaengig davon gemaess
[writing/README.md](writing/README.md) gebaut.

## Daten und erzeugte Artefakte

Rohdaten, aufbereitete Parquet-Dateien, Modell-Checkpoints, Reports und
LaTeX-Build-Artefakte koennen umfangreich sein und sind nicht zwingend Teil der
Versionsverwaltung. Die jeweiligen Unterprojekt-READMEs legen die erwarteten
Pfade und die reproduzierbar erzeugbaren Ausgaben fest.
