# WESPA - Web Service for Etherpad Analysis

## Installation

**Prerequesits**
- Python 3.12
- poetry

## How to use WESPA for analysis

- The data folder contains different database dumps of the polaris system
- Open the analysis.ipynb and either
  - run the processing form step 1 to step 5 or
  - run the "All at once" blocks at the lower part of the file
- Find the results in the output folder

## How to use WESPA as a webservice
WESPA can be used an analytics engine to process data from a etherpad database and sending the results on request.

## RQ EDM'25
- reproduce results for all datasets
- Ergebnisse deskriptiv aufbereiten
  - Woran kann man eine funktionale/dysfunktionale Gruppe erkennen?
  - Ab wann lässt sich erkennen, dass eine Gruppe nicht funktioniert?

- zeitlichen Verlauf der Netzwerkmaße darstellen
  - Hypothese: Gruppen finden unterschiedlich schnell zusammen

- schwerpunkt cohesion
  - Was ist group cohesion? Wie zeigt sie sich beim kollaborativen Schreiben? Wie kann man group cohesion beim kollaborativen Schreiben messen?
- cohesion maß mit anderen kollaborationsmaßen vergleichen
  - Besteht eine Zusammenhang zwischen cohesion und Umfang an Kommunikation?
  - Besteht ein Zusammenhang zwischen cohesion und Textqualität?
  - Welche zeitlichen Muster gibt es bei der Zusammenarbeit, die eine hohe/niedrige cohesion erklären?
- Qualität?
  - Einfachster Ansatz: Qualität nur am Ende der Arbeitsphase ermitteln.
  - in welchen zeitlichen Abständen kann man die Qualität bestimmen? => Nach jeder Session, doch was macht man bei synchroner Nutzung? => Nachweis, dass synchrones Handeln eine Ausnahme darstellt ODER: Qualitätszuwachs allen anwesenden gleichmaßen zuschreiben.
  - welche R-frameworks gibt es zur Bestimmung der Textqualität?
  - Wie lässt sich Qualität attribuieren?

**nth**
- window-size for authoring context
  - Was ist die optimale Fenstergröße, die den Zusammenhalt abbildet? Was könnte ein Referenzmaß für den Zusammenhalt sein?




## Unit Tests
python -m unittest tests.test.TestClass
