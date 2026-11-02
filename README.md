# LLM-Benchmarks

Lokale Sprachmodelle auf echter Hardware, gemessen mit einem Prompt für alle Geräte. Die Eingabe ist deutscher Fließtext mit 8.192 Tokens, die Antwort immer 256 Tokens lang. **Einlesen** zeigt, wie schnell ein Dokument verarbeitet ist, **Generierung**, wie schnell die Antwort entsteht.

## Ergebnisse

| Gerät | Modell | Klasse | Engine | Einlesen tok/s | Generierung tok/s | Datum |
|:------|:-------|:------:|:-------|---------------:|------------------:|:------|
| **PC RTX 3090 24 GB** | Qwen3.5 4B Q4_K_M | S | llama.cpp 0.7.0-dev (Build 11909) | 7.628 | 163,6 | 26.10.2026 |
| | Qwen3.5 4B Q4_K_M | S | Ollama 0.40.3 | 6.626 | 149,8 | 26.10.2026 |
| | Qwen3 14B Q4_K_M | M | Ollama 0.40.3 | 2.924 | 71,8 | 26.10.2026 |
| | Qwen3 14B Q4_K_M | M | llama.cpp 0.7.0-dev (Build 11909) | 2.782 | 71,4 | 26.10.2026 |
| | Qwen3.8 27B Q4_K_M | L | llama.cpp 0.7.0-dev (Build 11909) mit MTP | 1.507 | 67,0 | 26.10.2026 |
| | Qwen3.8 27B Q4_K_M | L | Ollama 0.40.3 | 1.351 | 66,4 | 26.10.2026 |
| | Qwen3.8 27B Q4_K_M | L | llama.cpp 0.7.0-dev (Build 11909) | 1.610 | 42,2 | 26.10.2026 |
| | Flint 40B-A2.5B Q4_K_M | F | llama.cpp 0.7.0-dev (Build 11909) | 3.183 | 121,7 | 28.10.2026 |
| | Flint 40B-A2.5B UD-Q3_K_XL | F | llama.cpp 0.7.0-dev (Build 11909) | 3.286 | 119,2 | 28.10.2026 |
| | Flint 40B-A2.5B NVFP4 | F | Ollama 0.40.12 (Eigenbau) | 2.512 | 117,0 | 28.10.2026 |
| | LFM3 1.7B Q4_K_M | Zusatz | llama.cpp 0.8.0-dev (Build 12001) | 18.882 | 307,4 | 30.10.2026 |
| | LFM3 1.7B Q4_K_M | Zusatz | Ollama 0.40.13 | 20.863 | 285,3 | 30.10.2026 |
| | Qwen3-Coder-30B-A3B Q4_K_M, abliteriert | Zusatz | llama.cpp 0.8.0-dev (Build 12001) | 4.408 | 162,4 | 30.10.2026 |
| | Qwen3.8-Flash-Next IQ3_S | Zusatz | Strata 0.2.8 (Engine 0.1.54) | 2.427 | 85,0 | 30.10.2026 |
| **MacBook Pro M5 Max 128 GB** | Qwen3.5 4B Q4_K_M | S | Ollama 0.40.11 | 4.171 | 118,0 | 26.10.2026 |
| **PC RTX 4060 Ti 8 GB** | Qwen3.5 4B Q4_K_M | S | llama.cpp b11889 (CUDA 13.4) | 5.420 | 72,2 | 26.10.2026 |
| | Qwen3.5 4B Q4_K_M | S | Ollama 0.40.11 | 4.938 | 59,4 | 26.10.2026 |
| | Qwen3 14B Q4_K_M | M | llama.cpp b11889 (CUDA 13.4) | passt nicht | passt nicht | 26.10.2026 |
| | Qwen3 14B Q4_K_M | M | Ollama 0.40.11 | passt nicht | passt nicht | 26.10.2026 |
| | Qwen3.8 27B Q4_K_M | L | llama.cpp b11889 (CUDA 13.4) | passt nicht | passt nicht | 26.10.2026 |
| | Qwen3.8 27B Q4_K_M | L | Ollama 0.40.11 | passt nicht | passt nicht | 26.10.2026 |
| **PC Radeon RX 560 4 GB** | Qwen3.5 4B Q4_K_M | S | llama.cpp 0.8.0-dev (Build 12024, Vulkan) | 136 | 20,4 | 31.10.2026 |
| | Qwen3.5 4B Q4_K_M | S | Ollama 0.40.13 | 135 | 18,3 | 31.10.2026 |

Klassen S bis L nutzen dieselbe Modellfamilie, F ist Flint als zweite Referenz, Zusatz liegt außerhalb der Standardklassen. „passt nicht“ heißt: Das Modell passt nicht vollständig in den GPU-Speicher.

## Benchmark-Prompt

Den Prompt in Claude Code oder einen vergleichbaren Agenten auf dem Testrechner einfügen und die Rückfragen beantworten. Vorher neu starten und möglichst alle anderen Programme beenden.

```text
Du führst einen standardisierten Benchmark für lokale Sprachmodelle auf diesem Rechner durch. Ziel: vergleichbare Messwerte für die Ergebnistabelle in https://github.com/maluramichael/llm-benchmarks. Arbeite die Schritte der Reihe nach ab. Nicht raten: Befehle, Flags und Modell-Tags vorher mit --help, der installierten Version oder der offiziellen Doku prüfen. Zuerst kommt eine Kurzanalyse mit Zeitschätzung, danach wählt der Nutzer aus, welche Tests laufen. Auch ein Teil der Tests ist ein wertvoller Beitrag.

SCHRITT 0: KURZANALYSE, ZEITSCHÄTZUNG UND AUSWAHL (vor allem anderen, nur lesen, höchstens 2 Minuten)
Der Nutzer soll vorab sehen, welche Tests auf diesem Rechner möglich sind und wie lange sie dauern, und selbst auswählen. Vor der Auswahl nichts installieren, herunterladen, beenden oder neu starten.
- Liegt unter ~/llm-benchmark/<heutiges Datum>/auswahl.json schon eine Auswahl (zum Beispiel weil der Prompt nach dem Neustart neu eingefügt wurde), diese mit der geschätzten Dauer zeigen und nur fragen, ob sie weiter gilt. Wenn ja, direkt mit VORBEREITUNG weitermachen.
- Ohne Rückfragen erfassen: Betriebssystem, Arbeitsspeicher, GPU und nutzbarer Speicher für Modelle (Befehle wie in SCHRITT 1 und ABSTURZSCHUTZ), freier Plattenplatz, installierte Engines mit Version (ollama --version, llama-server --version, auf Apple Silicon mlx-lm), schon vorhandene Modelle (ollama list, GGUF-Dateien im Hugging-Face-Cache und in üblichen Modellordnern).
- Daraus die möglichen Tests ableiten: jede Kombination aus Engine (SCHRITT 2) und Modellklasse (SCHRITT 3), die in den nutzbaren Speicher passt. Klassen, die nicht passen, gleich als "passt nicht" ausweisen, sie kosten keine Zeit.
- In der Ergebnistabelle des Repos nachsehen, welche Kombinationen es für diesen Gerätetyp schon gibt. Fehlende Kombinationen als "neu" markieren und in der Auswahl zuerst anbieten. Wiederholungen sind trotzdem erwünscht.
- Dauer pro Test als Spanne in Minuten schätzen:
  - Ein voller Test liest rund 75.000 Tokens ein (3 x 512 kalt, 3 x 8.192 warm, 6 x 8.192 Prompt-Cache) und erzeugt 3.072 Tokens (12 x 256), dazu kommt dreimal die Ladezeit. Der Kontext-Test liest jede Stufe einmal und die beste noch zweimal, 8k bis 128k sind zusammen rund 250.000 Tokens, bis 256k rund 500.000. Ab 64k mit halbem Einlesetempo rechnen. Auf langsamen Geräten kostet der Kontext-Test die meiste Zeit.
  - Einlesen und Generierung in tok/s aus der Ergebnistabelle nehmen: gleicher Gerätetyp, sonst das ähnlichste Gerät mit 30 Prozent Aufschlag. Ohne Vergleichswert vorsichtig schätzen und das dazuschreiben.
  - Vergleichswerte einer RTX 3090 mit 24 GB: ein voller Test (eine Engine, ein Modell, mit Kontext-Test) dauerte 6 bis 10 Minuten, llama-bench zusätzlich 2 bis 5 Minuten, das Zuschneiden der Texte einmal pro Modell 1 bis 2 Minuten.
  - Downloads getrennt ausweisen, mit Größe in GB und Dauer. Das Tempo kurz messen (etwa 15 Sekunden einer großen Hugging-Face-Datei mit curl laden, dann abbrechen), sonst 100 Mbit/s annehmen und das dazuschreiben. Vorhandene Modelle kosten keine Downloadzeit.
  - Einmalige Zeiten dazurechnen: Neustart und Aufräumen etwa 10 Minuten, jede fehlende Engine 5 bis 10 Minuten.
- Das Ergebnis als kurze Tabelle zeigen: Test (Engine, Modell, Klasse), neu oder schon gemessen, Download in GB, Dauer in Minuten als Spanne. Darunter eine Zeile "Alles zusammen: etwa X bis Y Minuten".
- Dann abfragen, was laufen soll (Mehrfachauswahl, jede Option nennt Dauer und Download):
  1. Modelle: die passenden Klassen zum Anhaken, neue Kombinationen zuerst und als empfohlen markiert.
  2. Engines: Ollama, llama.cpp, auf Apple Silicon zusätzlich MLX.
  3. Umfang pro Test: "Voll" (alle vier Messungen), "Ohne Kontext-Test" (Kaltstart, warm, Prompt-Cache, spart meist die Hälfte, Max. Kontext ist dann "nicht erfasst") oder "Nur warm" (3 Anfragen mit 8.192 Tokens, je Test etwa 1 bis 3 Minuten). Außerdem llama-bench ja oder nein.
  4. Zeitbudget, falls die Auswahl länger als 60 Minuten dauert: alles laufen lassen oder auf 30, 60 oder 120 Minuten kürzen. Beim Kürzen neue Kombinationen und kleine Modelle zuerst behalten.
- Die Auswahl mit der Gesamtdauer in einem Satz bestätigen lassen und als ~/llm-benchmark/<Datum>/auswahl.json speichern (Tests, Umfang, Schätzung pro Test). Alle folgenden Schritte gelten nur für diese Auswahl: nur die gewählten Engines prüfen oder installieren, nur die gewählten Modelle laden, nur die gewählten Messungen machen. Bei Umfang "Ohne Kontext-Test" oder "Nur warm" den Umfang in die Anmerkungen schreiben, fehlende Werte als "nicht erfasst".
- Während des Laufs nach jedem Test kurz Fortschritt und Restzeit melden (bisherige Dauer gegen Schätzung). Dauert ein Test mehr als doppelt so lange wie geschätzt, fragen, ob es weitergehen soll. Die tatsächliche Dauer pro Test in der JSON festhalten.

VORBEREITUNG (nach Schritt 0, vor Schritt 1)
Für die besten Ergebnisse muss so viel Arbeitsspeicher wie möglich frei sein. Auf Apple Silicon teilen sich CPU und GPU denselben Speicher, unter Windows belegen Browser, Docker und angeschlossene Monitore zusätzlich GPU-Speicher.
- Der Rechner muss frisch neu gestartet sein. Zeit seit dem letzten Start auslesen (macOS: sysctl -n kern.boottime, Windows: (Get-CimInstance Win32_OperatingSystem).LastBootUpTime). Liegt der Start mehr als 30 Minuten zurück, den Nutzer bitten, neu zu starten und den Prompt danach in einer neuen Session wieder einzufügen. Die Auswahl aus Schritt 0 bleibt in auswahl.json erhalten. Unter Windows "Neu starten" wählen, nicht Herunterfahren, sonst bleibt durch den Schnellstart der alte Zustand erhalten. Vorher die Punkte aus ABSTURZSCHUTZ prüfen, die selbst einen Neustart brauchen (TdrDelay), damit ein Neustart für alles reicht.
- So viele Programme wie möglich beenden, außer dem Agenten und seinem Terminal. Vor allem Colima, Docker (unter Windows danach auch wsl --shutdown), Browser, IDEs, Chat-Programme und Hintergrund-Apps in der Menüleiste bzw. im Infobereich (Cloud-Sync, Messenger, Updater). Laufende Programme nach Speicherverbrauch auflisten (macOS: ps -axo rss=,comm= | sort -rn | head -15, Windows: Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 Name,WorkingSet64) und den Nutzer bitten, sie zu beenden. Selbst beenden nur mit Zustimmung, weil ungespeicherte Arbeit verloren gehen kann.
- Wenn möglich alle externen Monitore abstecken, spätestens vor Schritt 4. Jeder Bildschirm belegt Speicher für Bildpuffer und Fensterdarstellung. Angeschlossene Monitore auflisten (macOS: system_profiler SPDisplaysDataType, Windows: Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams) und den Nutzer darauf hinweisen. Am Laptop bleibt nur das eingebaute Display an. Einen Desktop-PC ohne Monitor per SSH von einem anderen Gerät aus bedienen. Braucht er einen Monitor, diesen an die Onboard-Grafik anschließen (siehe ABSTURZSCHUTZ).
- Freien Arbeitsspeicher nach dem Aufräumen notieren. Wollte der Nutzer nicht neu starten, ein Programm weiterlaufen lassen oder einen Monitor angeschlossen lassen, das in den Anmerkungen jedes Tests festhalten (bei Monitoren: Anzahl und an welcher Grafik).

ABSTURZSCHUTZ (gilt für alle Schritte)
Läuft der GPU-Speicher voll, kann die GPU hängen bleiben und unter Windows einen Bluescreen auslösen. Beobachtet auf einer RTX 3090 beim Max-Kontext-Test von Qwen3.8 27B mit 32k Kontext. Deshalb:
- Windows mit NVIDIA: Den Nutzer vor dem Start bitten, in der NVIDIA-Systemsteuerung unter "3D-Einstellungen verwalten" die Option "CUDA - Sysmem Fallback Policy" auf "Prefer No Sysmem Fallback" zu stellen. Dann gibt es bei vollem Speicher einen sauberen Out-of-Memory-Fehler statt eines Hängers. Per Kommandozeile ist das nicht offiziell setzbar, also bestätigen lassen.
- Windows: Registry HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers auslesen. Fehlen TdrDelay und TdrDdiDelay (Standard 2 Sekunden), empfehlen, TdrDelay = 10 und TdrDdiDelay = 20 als DWORD zu setzen und neu zu starten. Nur mit Zustimmung ändern, TdrLevel nie auf 0 setzen.
- Bestenfalls hängt der Monitor an der Onboard-Grafik der CPU, dann rechnet die dedizierte GPU ohne Desktop und der Grafik-Watchdog greift kaum. Das geht nur mit iGPU: Intel ohne "F" im Modellnamen, AMD Ryzen bis Serie 5000 nur mit "G" (z. B. 5700G), ab Serie 7000 alle ohne "F". Prüfen und den Nutzer darauf hinweisen, aber nicht darauf bestehen.
- macOS: Die GPU darf nicht den ganzen Unified Memory nutzen. Die echte Grenze ist recommendedMaxWorkingSetSize von Metal. Auslesen: echo 'import Metal; print(MTLCreateSystemDefaultDevice()!.recommendedMaxWorkingSetSize / 1048576)' > /tmp/mtl.swift && swift /tmp/mtl.swift (Ergebnis in MiB), alternativ die Zeile recommendedMaxWorkingSetSize im llama-server-Log. Gemessen auf einem MacBook Pro M5 Max mit 128 GB: 110.100 MiB, also rund 107,5 GB (84 Prozent). Modell plus KV-Cache plus 1,5 GB Reserve müssen darunter bleiben, sonst den Test überspringen. sysctl iogpu.wired_limit_mb nur mit Zustimmung des Nutzers ändern (wirkt sofort, gilt bis zum Neustart, 0 heißt Standard), dabei mindestens 16 GB für macOS frei lassen und die Änderung in den Anmerkungen notieren.
- Vor jedem Test den freien GPU-Speicher prüfen (nvidia-smi --query-gpu=memory.used,memory.total --format=csv). Bleiben weniger als 1,5 GB frei, den Test überspringen und als "passt nicht" notieren.
- Windows: Nach jedem Test prüfen, ob seit Teststart ein GPU-Treiber-Reset passiert ist (System-Log: Quelle nvlddmkm Ereignis 153 oder Display Ereignis 4101, oder neue Datei unter C:\Windows\LiveKernelReports\WATCHDOG). Wenn ja, den ganzen Lauf sofort beenden, auch nicht mit kleineren Kontext-Stufen weitermachen. Der Bluescreen kann erst beim zweiten Hänger kommen. Den Nutzer um einen Neustart bitten und den Test danach mit Batch-Größe 256 wiederholen (Ollama num_batch, llama.cpp -ub), in den Anmerkungen notieren.
- Ergebnisse nach jedem einzelnen Test sofort als JSON speichern, nicht erst am Ende. Ein Absturz darf nur den laufenden Test kosten.

SCHRITT 1: SYSTEM ERFASSEN
Werte aus Schritt 0 übernehmen und nur Fehlendes ergänzen.
- Betriebssystem und Version, CPU, Arbeitsspeicher.
- GPU und nutzbarer Speicher für Modelle:
  - macOS: Chip und Unified Memory (system_profiler SPHardwareDataType, sysctl hw.memsize). Für Modelle nutzbar ist nur die Metal-Grenze recommendedMaxWorkingSetSize (siehe ABSTURZSCHUTZ), diese als "Speicher für Modelle" eintragen.
  - NVIDIA: nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  - AMD: rocm-smi bzw. unter Windows Get-CimInstance Win32_VideoController
- Freier Plattenplatz. Ist ein Laptop im Akkubetrieb, bitte das Netzteil anschließen lassen.
- Geräte-Identifier nach dem Muster "<Gerätetyp> <Chip oder GPU> <Speicher>" bilden, z. B. "MacBook Pro M5 Max 128 GB", "MacBook Air M1 16 GB", "PC RTX 3090 24 GB", und bestätigen lassen.

SCHRITT 2: ENGINES
Nur die in Schritt 0 gewählten Engines.
- Ollama (alle Systeme) und llama.cpp mit llama-bench und llama-server (alle Systeme). Auf Apple Silicon zusätzlich MLX (mlx-lm), falls der Nutzer zustimmt.
- Vorhandene Installationen nutzen. Fehlt etwas, vor der Installation fragen und nur offizielle Wege nehmen (Homebrew, winget, offizielle Releases bzw. Installer, für Python uv). Kein sudo ohne Rückfrage.
- Die Version jeder Engine notieren.

SCHRITT 3: MODELLE AUSWÄHLEN
Nur die in Schritt 0 gewählten Klassen.
Die Klassen S bis XL nutzen dieselbe Modellfamilie, damit die Geräte vergleichbar bleiben: Qwen3.x, Quantisierung Q4_K_M (MLX: 4 Bit). Dazu kommt Klasse F mit Flint als zweiter Referenz außerhalb der Qwen-Familie.
- Klasse S: kleinstes Qwen3.x-Modell um 4B Parameter.
- Klasse M: Qwen3.x-Modell um 14B Parameter.
- Klasse L: Qwen3.8 27B (das Referenzmodell).
- Klasse XL, nur bei mindestens 64 GB nutzbarem Speicher: das größte Qwen3.x-Modell, das mit mindestens 20 Prozent Reserve für den KV-Cache in den nutzbaren Speicher passt.
- Klasse F, nur bei mindestens 24 GB nutzbarem Speicher: Flint von Recursive (40B Parameter, Mixture of Experts mit rund 2,5B aktiven Parametern pro Token, bis 256k Kontext, KV-Cache nur rund 28 KiB pro Token). Dateien je Engine:
  - llama.cpp: Q4_K_M aus ggml-org/Flint-GGUF (Flint-Q4_K_M.gguf, 23,4 GB). Bei weniger als 32 GB nutzbarem Speicher zusätzlich UD-Q3_K_XL aus unsloth/flint-GGUF (flint-UD-Q3_K_XL.gguf, 18,5 GB): Auf einer 24-GB-Karte füllt Q4_K_M den Speicher schon bei 16k Kontext, die 3-Bit-Datei schafft 64k bei gleichem Tempo. Braucht einen llama.cpp-Build mit Flint-Unterstützung (getestet mit Build 11909, Commit caa8191df), ältere Builds vorher aktualisieren.
  - Ollama: flint:40b, das ist NVFP4, ein Q4_K_M gibt es in der Ollama-Library nicht. Braucht Ollama 0.40.9 oder neuer. Quantisierung mit ollama show prüfen und in den Anmerkungen nennen.
  - MLX: RecursiveSI/flint-int4 (INT4-Gewichte vom Hersteller, 22,6 GB). Fehlt die Flint-Unterstützung in der installierten mlx-lm-Version, MLX für Klasse F als "nicht erfasst" notieren.
  In Karte und Tabelle als Klasse F eintragen, z. B. "Flint 40B-A2.5B Q4_K_M (F)". Die Eingabetexte mit dem Tokenizer von Flint zuschneiden, er unterscheidet sich von Qwen.
Exakte Tags vorher in der Ollama-Library bzw. bei den GGUF-Dateien auf Hugging Face prüfen. Gibt es ein Modell nicht, das nächstliegende derselben Familie nehmen und das in den Anmerkungen festhalten. Eine Klasse, die nicht vollständig in den GPU-Speicher passt, überspringen und als "passt nicht" notieren. Vor jedem Download über 20 GB und bei knappem Plattenplatz nachfragen, außer der Download wurde in Schritt 0 schon mit Größe bestätigt.

SCHRITT 4: MESSEN
Eingabe ist deutscher Fließtext aus 17 Wikipedia-Artikeln (Rhein, Donau, Elbe, Hamburg, München und weitere), zusammen rund 1,8 MB. Den Text mit dem Tokenizer des jeweiligen Modells auf die gewünschte Länge zuschneiden (512 Tokens, 8.192 Tokens, beim Kontext-Test bis knapp an die Kontextgrenze) und die Aufgabe "Fasse den obigen Text in wenigen Sätzen sachlich zusammen" anhängen.
Für jede gewählte Kombination aus Engine und Modell, im gewählten Umfang. Feste Einstellungen: temperature 0, seed 42, 256 Ausgabe-Tokens, jede Messung 3 Mal, Median verwenden. Keine anderen rechenintensiven Programme parallel.
1. Kaltstart: Modell vollständig entladen (ollama stop bzw. Server neu starten), dann eine Anfrage mit 512 Tokens Eingabe. Messen: Ladezeit des Modells in Sekunden und Zeit bis zum ersten Token (TTFT kalt).
2. Warm: Modell bleibt geladen, neue Anfrage mit 8.192 Tokens Eingabe (neuer Text). Messen: TTFT warm, Einlesen in Tokens pro Sekunde, Generierung in Tokens pro Sekunde.
3. Prompt-Cache: zweimal derselbe 8.192-Token-Präfix mit jeweils anderer kurzer Frage am Ende. Messen: TTFT der zweiten Anfrage (zeigt, ob der Präfix-Cache greift).
4. Maximaler Kontext: Kontextgröße schrittweise erhöhen (8k, 16k, 32k, 64k, 128k, bis zum Maximum des Modells). Größter Wert, bei dem das Modell noch komplett auf der GPU läuft (Ollama: ollama ps zeigt 100% GPU) und eine Anfrage mit fast vollem Kontext durchläuft. Bei diesem Wert Einlesen und Generierung in Tokens pro Sekunde messen. Vorher pro Stufe abschätzen, ob Modell plus KV-Cache in den freien Speicher passen, und rechnerisch unpassende Stufen gar nicht starten. Die Stufenleiter sofort beenden, nie die nächste Stufe probieren, sobald eine Anfrage fehlschlägt, ollama ps weniger als 100% GPU zeigt, der geteilte GPU-Speicher (Windows: Leistungsindikator "GPU Process Memory", Shared Usage) gegenüber der vorherigen Stufe um mehr als 500 MB steigt, weniger als 1,5 GB GPU-Speicher frei bleiben oder unter macOS der Swap (sysctl vm.swapusage) um mehr als 1 GB wächst. Max. Kontext ist dann die letzte erfolgreiche Stufe.
Umfang "Ohne Kontext-Test": nur 1 bis 3. Umfang "Nur warm": nur 2.
Quellen für Messwerte: Ollama-API (load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration), llama-bench (Prompt- und Generierungstest, Tiefe über die passende Option laut --help), llama-server mit cache_prompt, mlx-lm laut --help. Tokenzahl der Eingabetexte mit dem Tokenizer des Modells prüfen.

SCHRITT 5: ERGEBNIS
Rohdaten als JSON unter ~/llm-benchmark/<Datum>/ speichern. Zahlen mit deutschem Dezimalkomma, fehlende Werte als "nicht erfasst".
a) Bericht als ~/llm-benchmark/<Datum>/ergebnis.md: Tabelle "System" (Gerät, Chip / GPU, Speicher für Modelle, Betriebssystem, Datum), Abschnitt "Tests" mit einer Karte pro Kombination aus Engine und Modell. Überschrift der Karte: "<Engine> <Version>, <Modell> <Quantisierung> (<Klasse>)". Felder in genau dieser Reihenfolge: Engine + Version, Modell (exakter Tag), Quantisierung, Klasse, Max. Kontext, Laden kalt (s), TTFT kalt (s), TTFT warm (s), TTFT mit Cache (s), Einlesen tok/s, Generierung tok/s, Generierung tok/s bei max. Kontext, Anmerkungen. Am Ende Abschnitt "Zusammenfassung" in drei Sätzen: bestes Setup auf diesem Gerät, Auffälligkeiten, Abbrüche. Keine Namen, Hostnamen, IP-Adressen oder Benutzerpfade in den Bericht schreiben.
b) Pro Test eine Zeile für die Ergebnistabelle ausgeben, Spalten in genau dieser Reihenfolge:
| Gerät | Modell | Klasse | Engine | Einlesen tok/s | Generierung tok/s | Datum |

SCHRITT 6: AUFRÄUMEN
Gestartete Server und Prozesse beenden. Heruntergeladene Modelle nur auf Nachfrage löschen.
```
