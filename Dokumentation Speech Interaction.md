# Speech Interaction: Dokumentation
## Wake Word Detection – “How Do You Wanna Do This?”

**Authors:** Julian Schniepp, Daniel Kling, Martin Herdt

# Inhaltsverzeichnis

[Projektidee](#projektidee)

[Wake Word Training](#wake-word-training)

[Positive Testdaten](#positive-testdaten)

[Negative Testdaten](#negative-testdaten)

[Offline Pipeline](#offline-pipeline)

[Methodik Evaluierung](#methodik-evaluierung)

[Technischer Ablauf der Auswertung](#technischer-ablauf-der-auswertung)

[Metriken und Validierungsgrundlage](#metriken-und-validierungsgrundlage)

[Code und Notebooks](#code-und-notebooks)

[Ergebnisse](#ergebnisse)

[Diskussion](#diskussion)

[“Hey Mycroft” Wake Word im Vergleich](#“hey-mycroft”-wake-word-im-vergleich)

[Deutung](#deutung)

[Mehrwortige Wake Words erleichtern Erkennung durch Timing-Komponente](#mehrwortige-wake-words-erleichtern-erkennung-durch-timing-komponente)

[Effektivität ist nicht linear zusammenhängend](#effektivität-ist-nicht-linear-zusammenhängend)

[Hoher Phoneme-Kontrast erleichtert Erkennung (Plosive, Konsonantengrenzen, Frikative)](#hoher-phoneme-kontrast-erleichtert-erkennung-(plosive,-konsonantengrenzen,-frikative))

[Mehr Daten == Besser, bei wenigen Daten ist Wake Word Wahl wichtig (Phonetik, Entropie)](#mehr-daten-==-besser,-bei-wenigen-daten-ist-wake-word-wahl-wichtig-(phonetik,-entropie))

[Layer Size kann Overfitting bewirken](#layer-size-kann-overfitting-bewirken)

[Anwendung](#heading=h.5eju7jiu87ge)

[Implementierung der „MercerAI“ Android-Applikation](#implementierung-der-„mercerai“-android-applikation)

[Zusammenfassung](#zusammenfassung)

[Reflexion](#reflexion)

[Ausblick](#ausblick)

[Quellen](#quellen)

# Projektidee

„How do you want to do this?“ ist eine Anspielung auf die populären Dungeons & Dragons-Kampagnen von *Critical Role (Critical Role Wiki, o. D.)*. Der Satz markiert den Moment, in dem ein Spieler den finalen Schlag gegen einen Gegner kreativ beschreiben darf. Für unser Projekt bietet dies zwei wesentliche Vorteile: Erstens einen konkreten, spielnahen Anwendungsfall und zweitens ein technisch anspruchsvolles, aber charakteristisches Wake Word.

# Wake Word Training

Aus technischer Sicht weist „How do you wanna do this?“ Merkmale auf, die sie deutlich von Standard-Wake Words (wie “Alexa” oder “OK Google”) abheben. Besonders relevant sind die hohe Silbenzahl, die Verwendung der umgangssprachlichen Form „wanna“, das markante prosodische Profil sowie die allgemeine Distinktivität. Diese Aspekte werden in den folgenden Abschnitten genauer beleuchtet.

Um die Auswirkungen der Phrasenlänge auf die Modellleistung und die Erkennungsrate empirisch untersuchen zu können, haben wir drei Varianten trainiert, die sich in ihrer Länge unterscheiden, aber akustisch ähnlich bleiben:

* **How\_L:** „How do you wanna do this“ (Langform / 7 Silben)  
* **How\_M:** „How you wanna do this“ (Mittellang / 6 Silben)  
* **How\_S:** „How you do this“ (Kurzform / 4 Silben)

Da in unserer Gruppe anfangs keine fundierten Python-Kenntnisse vorhanden waren, ergaben sich schnell erste fachliche Herausforderungen. Zudem war unser Vorwissen im Bereich KI weitgehend auf die Interaktion mit LLMs begrenzt. Auch die Nutzung des HdM-GPU-Clusters gestaltete sich schwieriger als erwartet: Aufgrund hartnäckiger Versionskonflikte und des damit verbundenen Debugging-Aufwands erwies sich Google Colab als die deutlich praktikablere Lösung für unser Zeitmanagement.  
Erschwerend kam hinzu, dass das offizielle Trainingsskript von OpenWake Word aufgrund seines Alters nicht mehr ausführbar war. Wir griffen stattdessen auf eine modifizierte Version des Nutzers LoresongGame zurück (LoresongGame, 2025). Während der Trainingsalgorithmus und die TTS-Engine dem Original entsprechen, wurden die Trainingsparameter angepasst, was eine erfolgreiche Ausführung in der aktuellen Umgebung ermöglichte.

## Ablauf Trainingsprozess

Das Training des Wake Word-Modells vollzog sich in drei zentralen Phasen, wobei wir auf die Infrastruktur von Google Colab zurückgriffen. Dies bot uns eine kostengünstige Umgebung mit vorinstallierten Abhängigkeiten, wodurch die zuvor aufgetretenen Versionskonflikte des lokalen Clusters gelöst werden konnten.

1. **Datengenerierung:** In dieser Phase wird mithilfe der PiperTTS-Engine ein vollständig synthetischer Datensatz erzeugt. Durch Variationen in der Sprechgeschwindigkeit und das Einfügen von Pausen wird sichergestellt, dass das Modell verschiedene Sprechweisen des Wake Words kennenlernt. Parallel dazu werden phonetisch ähnliche adversarielle Negativbeispiele generiert, indem Phoneme der Zielphrase systematisch variiert werden, um das Modell auf ähnlich klingende Phrasen vorzubereiten.   
2. **Augmentierung:** Um das Modell auf reale Bedingungen vorzubereiten, werden die Sprachdaten mit Hintergrundgeräuschen und Raumakustik unterlegt.  
3. **Modelltraining:** Das Modell ist als dreischichtiges Netzwerk aufgebaut, und trainiert in einer Google Colab-Umgebung. Dabei wird die Balance zwischen Erkennungsrate und Fehlaktivierungen (FRR/FAR) durch eine zunehmende Pönalisierung von Fehlern gesteuert und die Stabilität durch das Mitteln der besten Checkpoints optimiert.

## Steuerung und Parameter

Der Haupthebel zur Steuerung der FRR/FAR-Balance (False Rejection vs. False Acceptance Rate) lag in der Pönalisierung von Fehlaktivierungen, die im Verlauf des Trainings zunehmend verstärkt wurde. Bei der Konfiguration haben wir uns weitgehend an die empfohlenen Standardwerte gehalten:

| Parameter | Wert |
| :---- | :---- |
| Trainingsdaten | 30.000 Samples |
| Trainingsschritte | 30.000 Steps |
| False Activation Penalty | 2.000 |
| Layer Size | 96 |
| Target Recall, False Positive | 0,7 und 0,7 |

## Modellarchitektur

Das trainierte Modell basiert auf einem einfachen vollverbundenen neuronalen Netz (DNN). Als Eingabe dienen Speech Embeddings der Form (16 × 96), erzeugt durch das vorgelagerte, vortrainierte Embedding-Modell. Die 16 Zeitfenster mit je 96 Embedding-Dimensionen repräsentieren., 16 Frames mit je 96 Mel-Frequenzbändern, das zunächst auf einen Vektor der Länge 1.536 geflacht wird. Dieser wird durch drei lineare Schichten verarbeitet, zwischen denen jeweils eine LayerNorm-Normalisierung und eine ReLU-Aktivierungsfunktion eingesetzt werden. Die Ausgabe ist ein einzelner Wahrscheinlichkeitswert (Sigmoid) zwischen 0 und 1, der angibt, ob das Wake Word erkannt wurde. Bei einer Layer Size von 96 ergibt sich eine finale Modellgröße von ca. 650 KB im ONNX-Format, was den Einsatz auf ressourcenbeschränkten Geräten wie Mobiltelefonen ermöglicht. Dies entspricht quasi der gleichen Architektur, die in dem offiziellen OpenWake Word Repository vorgegeben wird.Erstellung der Testdaten

## Positive Testdaten

Zur Erstellung vieler Testdaten hat es sich angeboten, ein eigenes Jupyter-Notebook zu erstellen. Dieses sollte viele positive Testdaten mit passender Augmentation erstellen können.  
Hierzu bietet sich Kokoro TTS an. Dies war uns schon bekannt für seine hohe Leistungsfähigkeit auch bei Inferenz auf handelsüblichen Computern. Kokoro bietet einige Vorteile wie die Anpassungsfähigkeit der Stimmen durch verschiedene Parameter. Wir haben dabei hauptsächlich folgende Parameter zufallsgesteuert eingesetzt: Sprechgeschwindigkeit und Voice Blending, womit sich quasi eine unendliche Anzahl von Stimmen ergibt.  
Die Pipeline augmentiert zufällig Audiodaten mittels folgender Methoden:

- Lautstärke (Gain)  
- Low-Pass Filter für Distanzsimulierung  
- High-Pass Filter  
- Band-Pass Filter für "Telefon Effekt"  
- Pitch Shifting  
- Time Stretching  
- Room Impulse (Reverb und Convolution)  
- Hintergrundgeräusche

Die Stärke, in der die Effekte und deren Parameter angewendet werden, werden per Zufall in einem voreingestellten Intervall festgelegt.  
Dieser Ansatz erlaubt es uns eine Vielzahl einzigartiger Testdaten komplett automatisch zu generieren, die nicht mit PiperTTS generiert wurden. Durch die Seed-basierte Generation und JSON Manifest sind die Ergebnisse komplett nachvollziehbar und reproduzierbar.

## Negative Testdaten

Neben den positiven Testdaten war auch ein negativer Datensatz Bestandteil für die Evaluierung und den Vergleich der trainierten Wake Words. Ziel war es, eine realistische Klangkulisse zu schaffen, um die Widerstandsfähigkeit der Modelle gegenüber Fehlaktivierungen (False Acceptances) zu prüfen.

Die Erzeugung dieser Daten erfolgte automatisiert über ein entwickeltes Python-Skript. Dabei wurden zwei wesentliche Komponenten kombiniert:

* **Sprachanteil:** Als Basis dienten zufällige Textpassagen aus dem **LibriSpeech-Datensatz** (Random Text) (Panayotov, 2015).  
* **Störgeräusche:** Um eine natürliche Umgebung zu simulieren, wurden Hintergrundgeräusche aus dem DEMAND-Datensatz (Noise) (Thiemann, 2013) beigemischt.

Um eine konsistente Audioqualität sicherzustellen, orientiert sich die Erstellung des negativen Datensatzes an der Benchmarking-Methodik von Picovoice (2020). Die technische Umsetzung erfolgte in drei zentralen Schritten:

* **Akustische Varianz:** Für jedes Sprachsegment wurde ein zufälliger Startpunkt innerhalb der längeren Noise-Dateien gewählt, um eine hohe Vielfalt zu erzielen.  
* **Pegelanpassung:** Mittels RMS-Normierung (Root Mean Square) wurde ein festes Signal-Rausch-Verhältnis (SNR) von 10 dB eingestellt.  
* **Clippingschutz:** Um digitale Übersteuerungen durch die Signaladdition zu verhindern, wurde der Spitzenpegel des finalen Mixes automatisch auf -0,2 dB begrenzt.

Das Ergebnis ist ein Datensatz von 5 Stunden Laufzeit, exportiert im 16-Bit-PCM-WAV-Format. Dieser stellt eine Mischung aus zufälliger Sprache und Hintergrundrauschen dar und bietet somit eine passende Grundlage für den Vergleich der Wake Word-Modelle.

# Offline Pipeline

## Methodik Evaluierung

Um die Leistungsfähigkeit der trainierten Modelle objektiv zu bewerten, wurde die bestehende Applikation (`app.py`) um einen Offline-Modus erweitert (`app-offline.py)`. Dieser ermöglichte es, große Datenmengen automatisiert zu analysieren, ohne die Tests in Echtzeit durchführen zu müssen.

## Technischer Ablauf der Auswertung

Der Offline-Modus scannt ein definiertes Verzeichnis und bereitet jede Audiodatei durch eine Normalisierung auf 16 kHz (Mono) vor. Die eigentliche Detektionslogik spiegelt dabei exakt das Verhalten der Online-Anwendung wider, um die Vergleichbarkeit mit dem späteren Live-Betrieb zu gewährleisten:

* **Sliding-Window-Verfahren:** Die Audiodaten werden in festen Fenstergrößen (`chunk_size`) analysiert, wobei die Schrittweite über den Parameter `hop_ms` definiert wird.  
* **PeakPicker & Release:** Zur Detektion der Trigger wird ein PeakPicker-Algorithmus eingesetzt. Der implementierte Release-Wert sorgt dafür, dass das System nach einem erkannten Trigger für eine definierte Zeit gesperrt wird, um Mehrfachauslösungen desselben Ereignisses zu verhindern.  
* **Threshold-Sweep:** Um die Empfindlichkeit der Modelle zu ermitteln, wurde ein systematischer „Sweep“ über verschiedene Schwellenwerte durchgeführt. Die Untersuchung startete bei einem Threshold von 0.05 und wurde in inkrementellen Schritten von 0.05 bis zu einem Maximum von 0.95 gesteigert. Dieser granulare Ansatz erlaubt es, die gesamte Wahrscheinlichkeitsspanne des Modells abzudecken.

# Metriken und Validierungsgrundlage

Die Qualität der Wake Word-Erkennung wurde anhand zweier zentraler Metriken auf Basis der zuvor erstellten Datensätze bestimmt:

* **False Rejection Rate (FRR):** Zur Ermittlung der fälschlichen Abweisungen wurden 300 spezifische positive Samples verwendet. Die FRR gibt an, wie oft das System das tatsächliche Wake Word nicht erkannt hat.  
* **False Acceptance Rate (FAR):** Die Bewertung der Fehlaktivierungen erfolgte über den generierten negativen Datensatz (5 Stunden Laufzeit). Hierbei wurde gemessen, wie häufig das System fälschlicherweise einen Trigger auslöst, obwohl das Wake Word nicht gesprochen wurde.

# Code und Notebooks

In unserem Projekt haben wir einiges an selbst erstellten und bereits vorhandenen Notebooks verwendet, folgend ist eine Übersicht über das Wesentliche

| Titel | Zweck | Link | Kommentar |
| :---- | :---- | :---- | :---- |
| `openWake Word_trainer_fixed` | Training Notebook für unser Wake Word | [Github Repository](https://github.com/julian-schn/wakeword-howdoyouwanndothis/blob/main/openwakeword_trainer_fixed.ipynb) | Bereitgestellt von Reddit Nutzer u/LoresongGame [(Link)](https://www.reddit.com/r/speechtech/comments/1pfzucm/openwakeword_onnx_improved_google_collab_trainer/) |
| `automatic-model-trainer` | Offizielles Training Notebook für Wake Words von openWake Word | [Github Repository](https://github.com/julian-schn/wakeword-howdoyouwanndothis/blob/main/automatic-model-trainer.ipynb) | Nicht mehr funktionsfähig, wurde nicht eingesetzt |
| `threshold_sweep_analysis` | Analyse mit positiven Testdaten über verschiedene Thresholds für FRR | [Github Repository](https://github.com/julian-schn/wakeword-howdoyouwanndothis/blob/main/threshold_sweep_analysis.ipynb) | Gebaut für Google Colab mit Google Drive |
| `synthetic-positive-Wake Word-generator` | Notebook für automatische Erstellung von synthetischen positiven Testdaten | [Github Repository](https://github.com/julian-schn/synthetic-positive-wakeword-generator) | Gute Performance auch lokal, verwendet Kokoro TTS |
| Allgemeines Repository der Vorlesung | Erste Implementationen für den Pi, ELIZA | [Github Repository](https://github.com/julian-schn/113457a-speech_interaction) |  |
| MercerAI | Demo Android App mit Wake Word Erkennung, funktional in DnD Spielen | [Github Repository](https://github.com/daniwokl97/Mercer) |  |

# Ergebnisse

Die abschließende Analyse der trainierten Modelle erfolgte durch eine Gegenüberstellung der False Rejection Rate (FRR) und der False Acceptance Rate (FAR). Diese Metriken erlauben eine fundierte Aussage über die Balance zwischen Bedienbarkeit (Erkennungsrate) und Störfestigkeit (Vermeidung von Fehlalarmen).

![][image2]![][image3]

## Analyse der FRR

Die Untersuchung der FRR verdeutlicht, dass die Länge und die phonetische Komplexität des Wake Words maßgeblich die Erkennungsrate beeinflussen:

* **Referenzmodell:** Das Modell "Mycroft" weist über das gesamte Spektrum die niedrigste FRR auf. Sie startet bei ca. 34 % und steigt moderat auf knapp 60 % am oberen Ende des Messbereichs an.  
* **Längenvergleich:** Das Modell "How\_L" (Langform) verzeichnet die höchste FRR aller Testreihen. Bereits bei einem niedrigen Schwellenwert von 0,1 liegt die Fehlrate bei über 80 % und nähert sich bei strikteren Einstellungen der 100 %-Marke.  
* **Schnittpunkt-Phänomen:** Das Kurzmodell "How\_S" startet initial mit der zweitbesten Erkennungsrate (ca. 32 %), zeigt jedoch eine deutlich steilere Steigung als die Konkurrenzmodelle. Bei einem Schwellenwert von ca. 0,5 kreuzt die Kurve von "How\_S" die des Modells "How\_M", welches insgesamt eine stabilere, wenn auch relativ hohe FRR aufweist.

## Analyse der FAR

Die Messung der Fehlaktivierungen pro Stunde liefert ein differenziertes Bild zur Störfestigkeit der Modelle:

* **Robustheit der Kurzform:** Das Modell "How\_S" erzielt über fast den gesamten Messverlauf die geringste Fehlaktivierungsrate. Ab einem Schwellenwert von 0,35 stabilisiert sich die Kurve auf einem Plateau von lediglich 0,2 FA/h.  
* **Spitzenwerte:** Die höchste initiale Fehlrate zeigt "Mycroft" mit über 5 FA/h bei niedrigem Schwellenwert, geht jedoch ab einem Threshold von 0,75 gegen Null, einem Wert im oberen, restriktiven Bereich des Messspektrums.  
* **Performance von How\_M:** Das Modell "How\_M" weist insbesondere im mittleren Threshold-Bereich (0,1 bis 0,45) die schlechteste Performance auf und verbleibt länger auf einem höheren FA/h-Niveau als die Modelle "How\_L" und "How\_S".

![][image4]

## Analyse der DET-Kurve (Detection Error Tradeoff)

Die vorliegende DET-Kurve (Detection Error Tradeoff) dient zur Evaluation der Modellgüte, indem sie die FRR ins Verhältnis zu den Fehlalarmen pro Stunde (FA/h) setzt. Ein ideal arbeitendes System würde eine Kurve aufweisen, die sich so nah wie möglich an den Koordinatenursprung (unten links) anschmiegt.

* **Positionierung:** Das Modell "Mycroft" liegt am nächsten am Koordinatenursprung und bildet damit die Leistungsspitze ab.  
* **Kurvencharakteristik:** Das Modell "How\_S" weist einen markanten, fast vertikalen Verlauf auf. Während die Fehlaktivierungen nahezu konstant niedrig bleiben (nahe der 0,2 FA/h-Marke), variiert die FRR in diesem Bereich massiv zwischen 40 % und 85 %.  
* **Langformen:** Die Kurven von "How\_M" und "How\_L" sind deutlich nach oben (Richtung hoher FRR) verschoben. Beide Modelle erreichen erst bei Fehlaktivierungsraten oberhalb von 1,0 FA/h eine FRR von unter 70 %, was sie im direkten Vergleich zur Referenz und zur Kurzform schlechter positioniert.

# Diskussion

## “Hey Mycroft” Wake Word im Vergleich

OpenWakeWord bietet das vortrainierte Modell “Hey Mycroft” an, das wir im Vergleich für das Benchmarking verwendet haben. Hierbei handelt es sich nicht um einen fairen Vergleich, da das Mycroft Modell signifikant größere Trainingsdaten verwendet hat. Die Architektur ist die Gleiche, die folgenden Layers sind nennenswert verschieden

| Parameter | Hey Mycroft | How Do You Wanna Do This |
| :---- | :---- | :---- |
| Schichten | 3 | 3 |
| Layer Size | 64 | 96 |
| Wake Word | 2 Silben | 7 Silben |
| Positive Samples | \~100000 | 30000 |
| Adversarielle Negative | Nein | Ja |
| Modellgröße | \~0,41 MB | \~650 KB |
| Testdaten | 51 (manuell) | 300 (synthetisch) |

Trotz maßgeblicher Ungleichheit liefert der Vergleich wertvolle Erkenntnisse: Hey Mycroft demonstriert, was mit deutlich mehr Trainingsdaten und einem akustisch prägnanten Zweisilber erreichbar ist. Unsere Modelle bleiben in FRR und FAR klar dahinter zurück. Dies ist jedoch weniger ein Versagen der Architektur als ein erwartbares Ergebnis der geringeren Datenmenge und der höheren phonetischen Komplexität unserer Zielphrase. Der Vergleich dient daher weniger als absolute Leistungsmessung, sondern als Referenzpunkt zur Einordnung der Ergebnisse. 

## Deutung

### Mehrwortige Wake Words erleichtern Erkennung durch Timing-Komponente

Unsere Ergebnisse zeigen, dass How\_L trotz höherer FRR bei niedrigen Thresholds eine vergleichbar gute FAR wie Hey Mycroft erreicht. Diese niedrige FAR ist jedoch kritisch zu betrachten: Da How\_L 79–100% aller Audios ablehnt, werden zwangsläufig auch Fehlaktivierungen unterdrückt, ein Artefakt der schlechten Erkennungsrate, nicht ein Beweis für den Vorteil zeitlichen Kontexts. Mehrwortige Wake Words (z.B. „Hey Siri“, „OK Google“) haben in der Praxis oft Erkennungs­vorteile gegenüber Einwort‑Triggern, weil sie länger dauern, mehr zeitlichen Kontext liefern und damit Verwechslungen mit Spontansprache reduzieren. Die längere akustische Dauer und charakteristische Betonungs‑ bzw. Silbenmuster dieser Phrasen machen sie als akustische Sequenz seltener und besser von Hintergrundsprache und ähnlichen Wörtern zu trennen, was vor allem die False‑Alarm‑Rate senkt. Keyword‑Spotting‑Übersichtsarbeiten und Praxisberichte zu Wake‑Word‑Systemen beschreiben, dass kurze Phrasen oder mehrsilbige Wörter dem Modell erlauben, die Entscheidung über mehrere Frames und ggf. sogar etwas Audio nach der Phrase zu mitteln, was die Robustheit gegenüber Rauschen, Sprecher‑Variabilität und Timing‑Unschärfen verbessert (Chen et al.,2014). Gleichzeitig gilt aber als gut etabliert, dass Designfragen wie phonotaktische Auffälligkeit, Seltenheit im Alltagsdiskurs, klare Konsonanten‑Vokale‑Kontraste und genügend Trainingsdaten oft wichtiger sind als der bloße Unterschied „ein Wort vs. mehrere Wörter“, d.h. schlecht gestaltete Mehrwort‑Phrasen können trotz längerer Dauer weiterhin zu häufigen Fehlaktivierungen führen

### Effektivität ist nicht linear zusammenhängend

In unseren Messungen zeigt sich dieser nichtlineare Zusammenhang deutlich: How\_M (6 Silben) schneidet in der DET-Kurve über weite Strecken schlechter ab als sowohl How\_S (4 Silben) als auch How\_L (7 Silben), eine moderate Verlängerung brachte hier keinen Vorteil, sondern die schlechteste FAR aller drei Varianten. Die Forschung zu Keyword Spotting und Wake Words legt nahe, dass der Zusammenhang zwischen Trigger‑Länge und Erkennungsleistung deutlich nichtlinear ist: Sehr kurze Keywords (ein bis zwei Silben) neigen zu erhöhten False‑Accept‑Raten, weil ihre akustischen Muster oft mit Alltagswörtern und Hintergrundsprache kollidieren, während eine moderate Verlängerung (mehrsilbige Wörter bzw. kurze Phrasen wie „Hey Siri“, „OK Google“) typischerweise klar messbare Verbesserungen bei Detektionsgenauigkeit und Robustheit gegenüber Sprecher‑Variabilität und Ausspracheunterschieden bringt  (van Leeuwen et al., 1999). Jenseits dieser moderaten Länge zeigen Studien jedoch ausgeprägte „diminishing returns“: Zusätzliche Silben oder Wörter liefern kaum noch diskriminative Information, erhöhen aber die Wahrscheinlichkeit verkürzter oder variierter Realisierungen durch Nutzer, was die False‑Reject‑Rate steigen lassen kann und die Modelle stärker belastet. In der Literatur wird daher ein Designkorridor empfohlen, in dem das Wake Word lang und phonetisch reichhaltig genug ist, um sich klar von Spontansprache abzusetzen, aber kurz genug bleibt, damit Sprecher es konsistent aussprechen können und die Modellkapazität nicht auf unnötig lange Sequenzen verteilt wird. Jenseits dieses Korridors überwiegen typischerweise Bedienbarkeits‑ und Robustheitsprobleme gegenüber weiteren Erkennungsgewinnen (López-Espejo et al., 2022).

### Hoher Phoneme-Kontrast erleichtert Erkennung (Plosive, Konsonantengrenzen, Frikative)

Der Leistungsvorsprung von Hey Mycroft über alle Thresholds lässt sich nicht allein auf die größere Trainingsdatenmenge zurückführen: Die harte Konsonantenfolge in „croft" bietet eine akustische Prägnanz, die unsere Phrasen mit dem vokalreichen „wanna" und weichen Endlauten so nicht aufweisen. In der Wake‑Word‑ und Keyword‑Spotting‑Literatur gilt als relativ gut etabliert, dass ein hoher phonemischer Kontrast und deutlich segmentierte Konsonanten–Vokal‑Strukturen die Erkennbarkeit verbessern, weil sie das akustische Signal klarer von Hintergrundsprache und ähnlich klingenden Wörtern absetzen und so sowohl Detektionsgenauigkeit erhöhen als auch False‑Reject‑Raten senken. Harte Konsonanten wie Plosive (z.B. „k", „t", „p") und Frikative (z.B. „s", „f") (Wang et al., 2024) erzeugen im Audiosignal klare, markante Muster, die ein Modell leichter erkennt. Rein vokalreiche Sequenzen sind dagegen anfälliger für Verschleifung, je nach Sprecher, Dialekt oder Sprechtempo klingen sie schnell ähnlich. Daher empfiehlt es sich, Wake Words mit deutlichen Konsonanten und klarer Silbenstruktur zu wählen. Ein universelles Rezept gibt es allerdings nicht, da die optimale Lautgestalt immer auch vom Modell und den Trainingsdaten abhängt.

### Mehr Daten == Besser, bei wenigen Daten ist Wake Word Wahl wichtig (Phonetik, Entropie)

Mit nur 30.000 positiven Samples gegenüber \~100.000 bei Hey Mycroft sind unsere Modelle ein direktes Beispiel für ein Low-Resource-Szenario, die deutlich höheren FRR-Werte von How\_L und How\_M legen nahe, dass die phonetische Komplexität unserer Phrasen bei dieser Datenmenge nicht ausreichend abgedeckt werden konnte. In der Wake‑Word‑Forschung gilt als gut etabliert, dass mehr und vielfältige Trainingsdaten die Erkennungsleistung meist deutlich verbessern (Jia et al., 2020): Mit wachsendem Datenvolumen sinken Fehler­raten (False Rejects und False Accepts), weil das Modell mehr Sprecher, Umgebungen und Aussprachevarianten abdeckt und dadurch robuster generalisiert. Gerade bei kleinen Datensätzen zeigt sich jedoch, dass die intrinsische „Erkennbarkeit“ der Zielphrase (also ihre phonetische Auffälligkeit, klare Segmentstruktur und hohe akustische Informationsdichte) überproportional wichtig wird, weil das Modell weniger Gelegenheit hat, systematische Ambiguitäten oder ungünstige Lautmuster aus den Daten herauszulernen. Studien zu „limited training data“ und user‑definierten Keywords berichten, dass in Low‑Resource‑Szenarien deutlich häufiger auf Datenaugmentation, synthetische Daten und explizit phonetisch günstige Keywords zurückgegriffen werden muss (Jia et al., 2020); andernfalls steigen insbesondere die False‑Reject‑Raten stark an, während bei reichlich Trainingsdaten die konkrete Lautgestalt des Keywords zwar weiterhin relevant bleibt, aber ein Teil der Defizite durch schiere Datenmenge kompensiert werden kann. Dieser Ressourcenmangel erklärt auch, warum die Erkennungsrate von How\_L bereits bei mittleren Schwellenwerten (Threshold \> 0,8) fast vollständig einbricht. Bei einer 7-silbigen Phrase müssen alle akustischen Teilsegmente die Schranke passieren; statistisch gesehen sinkt die Wahrscheinlichkeit für eine solche ‚perfekte Kette‘ bei strengen Grenzwerten gegen Null, wenn das Modell aufgrund der geringen Datenmenge keine ausreichende Konfidenz für die gesamte Sequenz entwickeln konnte.

### Layer Size kann Overfitting bewirken

Die Modelle zeigen im Threshold-Sweep teils instabile Kurvenverläufe, die auf mangelnde Generalisierung hindeuten. Als Hauptursache ist weniger die Layer Size von 96 zu sehen, sondern vielmehr die Kombination aus einer sehr hohen False Activation Penalty (2.000) und der geringen Sprachvarianz der rein synthetischen PiperTTS-Trainingsdaten: Die Verlustfunktion trainiert das Modell extrem aggressiv gegen Fehlaktivierungen, während die Synthetik-Daten die natürliche Varianz menschlicher Stimmen nur begrenzt abdecken. In der Forschung zu neuronalen Netzen, insbesondere bei Low-Data-Szenarien, ist es ein gut etabliertes Prinzip, dass Modelle mit zu hoher Kapazität (z. B. zu viele Schichten oder Parameter relativ zur Trainingsdatenmenge) stark zum Overfitting neigen: Sie merken Trainingsbeispiele auswendig, anstatt generalisierbare Muster zu lernen, was zu hohen Trainingsgenauigkeiten, aber schlechter Generalisierung auf neuen Daten führt. Dies tritt besonders bei kleinen Datensätzen auf, da die Modellkomplexität die Datenvielfalt übersteigt und Rauschen oder Ausreißer priorisiert werden; kleinere, einfachere Modelle sind hier robuster und erzwingen komprimierte Repräsentationen mit besserer Vorhersagekraft. Als grobe Daumenregel gilt in der Praxis ein Verhältnis von etwa 10–100 Trainingsbeispielen pro Parameter (je nach Aufgabe und Regularisierung), wobei speziell in embedded KWS- und Small-Footprint-Forschung (z. B. für mobile Keyword-Spotting) Modelle mit \<1 Mio. Parametern für begrenzte Daten priorisiert werden, um Overfitting zu vermeiden – oft kombiniert mit Dropout, Data Augmentation oder Transfer Learning. Ein deutliches Indiz für diese Überoptimierung ist die markante Steilheit der DET-Kurve bei How\_S. Der fast vertikale Verlauf zeigt, dass die Fehlaktivierungen zwar extrem niedrig bleiben, die Erkennungsrate (FRR) aber bei minimaler Erhöhung des Thresholds sofort massiv einbricht. Das Modell hat eine ‚starre' Entscheidungsgrenze gelernt, die zwar effektiv vor Fehlalarmen schützt, aber kaum Spielraum für natürliche Aussprachevariationen lässt

# Anwendung

## Implementierung der „MercerAI“ Android-Applikation

Die mobile Applikation „MercerAI“ ([https://github.com/daniwokl97/Mercer](https://github.com/daniwokl97/Mercer)) dient als sprachgesteuerter Kampf-Assistent für das Rollenspiel Dungeons & Dragons. Sie ermöglicht es, kreative Kampfchoreografien durch Spracheingabe zu erkennen und diese unmittelbar akustisch zu untermalen.

1\. ONNX-Runtime in Android

Die lokale Ausführung der Wake Word-Erkennung basiert auf der Library openWake Word-android-kt (Re-MENTIA, 2024), einer Kotlin-Portierung des OpenWake Word-Frameworks. Sie ermöglicht eine hocheffiziente Inferenz mittels ONNX Runtime direkt auf dem Endgerät.

* **Parallele Inferenz:** Für die gleichzeitige Überwachung der drei Modelle („How\_L“, „How\_M“, „How\_S“) wird die `ParallelWake WordEngine` genutzt. Ein Worker-Pool führt die Berechnungen unabhängig voneinander aus, was die Reaktionszeit bei mehreren aktiven Wake Words massiv senkt.  
* **Automatisierte Signalverarbeitung:** Die Library abstrahiert die Vorbereitung der Audiodaten. Unter Verwendung von Basis-Modellen (`melspectrogram.onnx` und `embedding_model.onnx`) im Assets-Ordner werden Rohdaten automatisiert in Mel-Spektrogramme und Speech-Embeddings transformiert.  
* **Effizienz & Datenschutz:** Durch die Nutzung von Kotlin Coroutines erfolgt die Verarbeitung non-blocking im Hintergrund. Da alle Berechnungen lokal erfolgen, bleiben Audiodaten privat und das System ist vollständig offline-fähig.

2\. Voice-Processing

1. **Detektion:** Die `CombatActivity` lauscht kontinuierlich auf die Signalwörter. Um Mehrfachtrigger während einer aktiven Verarbeitung zu vermeiden, wird der Zugriff über einen `AtomicBoolean` (`isBusy`) und eine zeitliche Sperre (`systemIsReadyToListen`) geschützt.  
2. **Ducking & Feedback:** Bei Erkennung eines Triggers (basierend auf den evaluierten Thresholds) stoppt die Engine kurzzeitig. Ein akustisches Signal („ack“) signalisiert Bereitschaft, während gleichzeitig die Hintergrundlautstärke via Audio-Ducking gesenkt wird.  
3. **Spracherkennung (STT):** Ein `SpeechToTextWrapper` nutzt den nativen Android `SpeechRecognizer`, um die darauffolgende Kampfansage des Nutzers in Text zu wandeln.

3\. Gemini-LLM

Zur Interpretation freier Sprachbefehle (z. B. „Ich schlage dreimal mit dem Schwert zu“) ist das Gemini-Flash-Modell (`gemini-2.5-flash`) integriert.

* **Intent-Parsing:** Die KI fungiert als Kampf-Logik-Modul. Anhand einer System-Instruction und eines hinterlegten „Handbuchs“ extrahiert sie Parameter wie Waffen-ID, Anzahl der Angriffe und Trefferzonen (`head`, `body`, `squishy`).  
* **Strukturierte Ausgabe:** Das Ergebnis wird als JSON-Objekt (`CombatAction`) ausgegeben, welches direkt von der App-Logik zur Steuerung der Sound-Events genutzt werden kann.

4\. Audio via FMOD (Native C++)

Die akustische Umsetzung erfolgt über die FMOD Audio Engine, die zur Minimierung von Latenzen über ein natives C++ Script angebunden ist.

* **JNI-Integration:** Die Kommunikation erfolgt über das Java Native Interface (JNI). Dies ermöglicht eine performante Verarbeitung der Audio-Events bei gleichzeitig geringer CPU-Last auf der Android-Ebene.  
* **Dynamisches Sound-Management:** Das C++ Script steuert über FMOD-Busse das Echtzeit-Ducking und den Wechsel zwischen Umgebungs- und Kampfszenarien. Nach Abschluss der Kampfhandlungen stellt das System automatisch den Normalzustand wieder her und aktiviert die Wake Word-Engine erneut.

 ![][image5]  
Screenshot von MercerAI - Work in Progress

# Zusammenfassung

Im Rahmen dieses Projekts wurde mit „MercerAI“ ein funktionaler Prototyp eines sprachgesteuerten Dungeons & Dragons-Kampfassistenten entwickelt. Der Kern des Systems liegt in der Implementierung eines benutzerdefinierten Wake Words („How do you want to do this?“), das in drei Längenvariationen (How\_L, How\_M, How\_S) trainiert und evaluiert wurde.

Durch den Einsatz einer rein synthetischen Trainingspipeline (PiperTTS) und einer diversifizierten Testumgebung (Kokoro TTS) konnte trotz begrenzter Ressourcen ein funktionales Modell erstellt werden. Die technische Umsetzung erfolgte über kompakte Architektur zur Signalverarbeitung, das mittels ONNX-Runtime effizient und datenschutzkonform lokal auf Android-Endgeräten ausgeführt wird.

Die Ergebnisse verdeutlichen das Spannungsfeld zwischen Phrasenlänge und Erkennungsrate: Während die Kurzform How\_S eine beeindruckende Störfestigkeit gegenüber Fehlaktivierungen (FAR) aufweist, führt die phonetische Komplexität der Langform How\_L zu einer sehr hohen Ablehnungsrate (FRR). 

# Reflexion

Konkrete Schritte, die wir bei zukünftiger Weiterarbeit vornehmen würden, beziehen sich primär auf das Training bei der Layer Size und der Anzahl der Ebenen selbst. Angesichts unserer begrenzten Trainingsdatenmenge wäre ein erster naheliegender Schritt, mit kleineren Layer Sizes (z.B. 32 oder 64) zu experimentieren, um Overfitting gezielt zu reduzieren. Gleichzeitig haben wir während des Projekts gemerkt, dass wir die Architektur weitgehend als gegeben hingenommen haben, hier würden wir beim nächsten Mal bewusst Zeit einplanen, um alternative Ansätze zu recherchieren und zu evaluieren, beispielsweise CNN-basierte oder attention-basierte Modelle, die auch für Keyword Spotting diskutiert werden. Das einfache DNN hat für den Einstieg gut funktioniert, aber wir sind gespannt, ob andere Architekturen unsere spezifische Herausforderung, ein langes, phonetisch variables Wake Word mit wenig Trainingsdaten, besser lösen könnten.

# Ausblick

Das Projekt bietet verschiedene Anknüpfungspunkte für zukünftige Optimierungen und Erweiterungen:

* **Diversifizierung der Trainingsdaten:** Ein entscheidender nächster Schritt wäre die Anreicherung des Datensatzes durch reale Sprachaufnahmen (Crowdsourcing), um die Varianz menschlicher Stimmen und Dialekte über die synthetischen Möglichkeiten hinaus abzudecken und so die FRR weiter zu senken.  
* **Feinjustierung der Modellkapazität:** Um das beobachtete Overfitting bei kleinen Datensätzen zu minimieren, könnten Experimente mit reduzierten Layer Sizes oder verstärkten Regularisierungstechniken (wie Dropout) durchgeführt werden. Ziel ist eine stabilere DET-Kurve, die weniger sensibel auf Threshold-Schwankungen reagiert.  
* **UX-Studie im Tabletop-Szenario:** Ein entscheidender nächster Schritt ist die Durchführung einer empirischen Nutzerstudie direkt am Spieltisch. Hierbei soll untersucht werden, wie sich die Sprachsteuerung auf den Spielfluss und die Immersion auswirkt. Zentrale Fragen sind hierbei: Wird das Wake Word als natürliche Interaktion empfunden oder stört es die narrative Dynamik? Wie reagieren Spieler auf das akustische Feedback in einer realen Geräuschkulisse?

# Quellen

Chen, G., Parada, C., & Heigold, G. (2014). Small-footprint keyword spotting using deep neural networks. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 4087–4091. [https://doi.org/10.1109/ICASSP.2014.6854370](https://doi.org/10.1109/ICASSP.2014.6854370)

Critical Role Wiki. (o. D.). *How do you want to do this?* Fandom. Abgerufen am 22\. Februar 2026 von [https://criticalrole.fandom.com/wiki/How\_do\_you\_want\_to\_do\_this%3F](https://criticalrole.fandom.com/wiki/How_do_you_want_to_do_this%3F)

dscripka. (2023). *openWakeWord* \[Computer-Software\]. GitHub. [https://github.com/dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)

hexgrad. (2024). *Kokoro-82M* \[Computer-Software\]. Hugging Face. [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

Jia, Y., Cai, Z., Ma, M., Zhao, Z., Wang, X., Wang, J., & Li, M. (2020). *Training wake word detection with synthesized speech data on confusion words*. arXiv. [https://doi.org/10.48550/arXiv.2011.01460](https://doi.org/10.48550/arXiv.2011.01460)

López-Espejo, I., Tan, Z.-H., Hansen, J. H. L., & Jensen, J. (2022). Deep spoken keyword spotting: An overview. *IEEE Access, 10*, 4169–4199. [https://doi.org/10.1109/ACCESS.2021.3139508](https://doi.org/10.1109/ACCESS.2021.3139508)

LoresongGame. (2025, 2\. September). *OpenWake Word ONNX Improved Google Collab Trainer* \[Online-Forumspost\]. Reddit. [https://www.reddit.com/r/speechtech/comments/1pfzucm/openwakeword\_onnx\_improved\_google\_collab\_trainer/](https://www.reddit.com/r/speechtech/comments/1pfzucm/openwakeword_onnx_improved_google_collab_trainer/)

Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). Librispeech: An ASR corpus based on public domain audio books. *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5206–5210. IEEE. [http://www.openslr.org/12/](http://www.openslr.org/12/)

Picovoice. (2020). *Wake Word Detection Benchmark*. https://picovoice.ai/docs/benchmark/wake-word/

Re-MENTIA. (2024). *openWake Word-android-kt: Kotlin library that brings openWake Word to Android* \[Computer-Software\]. GitHub. [https://github.com/Re-MENTIA/openWakeWord-android-kt](https://github.com/Re-MENTIA/openWakeWord-android-kt)

Rhasspy. (2023). *Piper: A fast, local neural text to speech system* \[Computer-Software\]. GitHub. [https://github.com/rhasspy/piper](https://github.com/rhasspy/piper)

Thiemann, J., Ito, N., & Vincent, E. (2013). *DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments* (Version 1.0) \[Datensatz\]. 21st International Congress on Acoustics (ICA 2013), Montreal, Kanada. Zenodo. [https://doi.org/10.5281/zenodo.1227121](https://doi.org/10.5281/zenodo.1227121)

van Leeuwen, D. A., Kraaij, W., & Ekkelenkamp, R. (1999). Prediction of keyword spotting performance based on phonemic contents. *Proceedings of ESCA Workshop Accessing Information in Spoken Audio*, 79–82. [https://www.isca-archive.org/accessaudio\_1999/leeuwen99\_accessaudio.pdf](https://www.isca-archive.org/accessaudio_1999/leeuwen99_accessaudio.pdf)

Wang, J., Li, X., Zhang, Y., Chen, L., & Liu, H. (2024). *Phoneme-level contrastive learning for user-defined keyword spotting with flexible enrollment*. arXiv. [https://arxiv.org/abs/2412.20805](https://arxiv.org/abs/2412.20805)

Warden, P. (2018). *Speech commands: A dataset for limited-vocabulary speech recognition*. arXiv. [https://doi.org/10.48550/arXiv.1804.03209](https://doi.org/10.48550/arXiv.1804.03209)

 

[image2]: assets/image2.png

[image3]: assets/image3.png

[image4]: assets/image4.png

[image5]: assets/image5.png
