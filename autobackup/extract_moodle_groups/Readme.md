# GroupGrep

Ein Skript um alle Lerngruppen eines Semesters in Form einer groups.json aus Moodle zu extrahieren.

1. Benutzername des Moodle-Kontos bei `mdl_username` eingeben.
2. Passwort des Moodle-Kontos bei `mdl_password` eingeben.
3. Eindeutige ID des gewünschten Kurses bei `mdl_courseid` eingeben. Die Kurs-ID steht in der URL des Kurses. [Moodle-Instanz](https://moodle-ddll.fernuni-hagen.de/login/index.php)
4. Script ausführen: `python3.9 groupgrep.py`
5. Liste auf den Server unter `/docker/backup` kopieren, z.B. `scp ./groups.json polaris://docker/backup/`
6. Die `courseid` bei .env unter `/docker/backup` auf den gewünschten Kurs anpassen.
7. `docker compose up --force-recreate --build` unter `/docker/backup` ausführen.
8. Backup-Daten liegen im Ordner `/docker/backup/backup` in einem eigenen Ordner mit passenden UNIX-Zeitstempel:

---

Error: Could not get instanceId of projectId 2 and groupId 4675
cwe_backup | at /home/node/app/index.js:59:43
cwe_backup | at processTicksAndRejections (node:internal/process/task_queues:96:5)

Error: Could not get instanceId of projectId 2 and groupId 4691
cwe_backup | at /home/node/app/index.js:59:43
cwe_backup | at runMicrotasks (<anonymous>)
cwe_backup | at processTicksAndRejections (node:internal/process/task_queues:96:5)
