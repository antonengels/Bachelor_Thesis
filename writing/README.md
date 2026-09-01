# Bachelorarbeit: Writing

Dieser Ordner enthält den LaTeX-Quelltext für die schriftliche Ausarbeitung der Bachelorarbeit **„Neuronale Modellierung realer Fahrstrategien im Schienenverkehr auf Basis historischer Betriebsdaten“**.

## Voraussetzungen

- Eine aktuelle LaTeX-Distribution, zum Beispiel [TeX Live](https://www.tug.org/texlive/) oder [MiKTeX](https://miktex.org/)
- `latexmk`
- `biber` für das Literaturverzeichnis
- `makeglossaries` für das Abkürzungsverzeichnis

Die verwendeten LaTeX-Pakete müssen in der Distribution installiert sein. MiKTeX kann fehlende Pakete bei Bedarf automatisch nachinstallieren.

## PDF erzeugen

Im Verzeichnis dieses Projekts ausführen:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Das fertige Dokument wird unter `out/main.pdf` abgelegt. `latexmk` übernimmt dabei die notwendigen Folgekompilierungen sowie die Verarbeitung des Literatur- und Abkürzungsverzeichnisses.

Zum Entfernen der erzeugten Hilfsdateien:

```powershell
latexmk -C main.tex
```

## Projektstruktur

```text
writing/
├── main.tex                         # Haupteinstiegspunkt und Dokumentaufbau
├── literature.bib                   # Literaturdatenbank
├── chapters/                        # Kapitel der Bachelorarbeit
│   ├── 01_einleitung.tex
│   ├── 02_theoretische_grundlagen.tex
│   ├── 03_stand_der_technik.tex
│   ├── 04_datenbasis_preprocessing.tex
│   ├── 05_modellarchitektur.tex
│   ├── 06_implementierung_training.tex
│   ├── 07_evaluierung_ergebnisse.tex
│   └── 08_fazit_ausblick.tex
├── frontmatter/                     # Präambel, Titelseite und Verzeichnisse
│   ├── abstract.tex
│   ├── acronyms.tex
│   ├── erklaerung.tex
│   ├── indizes.tex
│   ├── listings-style.tex
│   ├── settings.tex
│   └── titlepage.tex
├── images/                          # Logos, Diagramme und weitere Abbildungen
└── out/                             # Generierte Build-Ausgabe
```

## Arbeiten am Dokument

- Für Änderungen am Aufbau oder an der Reihenfolge der Bestandteile ist `main.tex` zuständig.
- Inhaltliche Änderungen werden im jeweils passenden Kapitel unter `chapters/` vorgenommen.
- Abkürzungen werden in `frontmatter/acronyms.tex` gepflegt.
- Formelgrößen und Einheiten werden in `frontmatter/indizes.tex` gepflegt.
- Literaturquellen werden in `literature.bib` ergänzt und im Text mit `biblatex` zitiert.
- Abbildungen werden in `images/` abgelegt und aus den Kapiteln referenziert.

## Hinweise

- Der Build muss aus dem Projektwurzelverzeichnis ausgeführt werden, damit alle relativen Pfade korrekt aufgelöst werden.
- Nach Änderungen an Literatur- oder Glossareinträgen sollte das Dokument vollständig neu gebaut werden.
- Temporäre LaTeX-Dateien gehören nicht in die Versionsverwaltung; die erzeugte PDF liegt im Ausgabeordner `out/`.
