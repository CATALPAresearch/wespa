# WESPA - Web Service for Etherpad Analysis


# todo
* Optimize: extract_neigbors which eats up most of the processing time
* summarize communication data to author...csv
* Anforderung von Jennifer:
    Wir brauchen für das WS 23/24 auf individueller Ebene 
    - Degree Centrality 
    - Closeness Centrality 
    für die Zeiträume 
    - 30.10. - 19.11.
    - 20.11. - 10.12.
    - 30.10. - 10.12.
* implement thread in the webservice
* collect text progress by session
* write db load  ---------- https://github.com/mburchart/cwt_import
    Mögliche SSH-Config:
    Host polaris
    HostName polaris.fernuni-hagen.de
    PreferredAuthentications publickey
    user niels
    IdentityFile [...]
    LocalForward 7000 127.0.0.1:5984

    Unter Port 7000 kannst du dann die CouchDB erreichen.
    

## RQ EDM'25
## Ideen für EDM
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




# Tests
python -m unittest tests.test.TestClass
