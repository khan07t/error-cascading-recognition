"""Invented plant layout and fault messages for the generated alarm log.

All station names and message texts are made up for this repository. Only the
column layout of the alert export is taken from the original project.
"""

from dataclasses import dataclass

LINES = ["A1", "A2", "B1"]

# Station roles in process order along a line.
ROLES = [
    ("PRS", "Pressen"),
    ("SWE", "Schweissen"),
    ("ROB", "Roboterzelle"),
    ("KLB", "Kleben"),
    ("FOE", "Foerdertechnik"),
    ("PUF", "Puffer"),
    ("FTS", "Fahrerloses Transportsystem"),
    ("MON", "Montage"),
    ("MES", "Messzelle"),
    ("PRF", "Pruefstand"),
]


@dataclass(frozen=True)
class Family:
    key: str
    messages: tuple              # (device code, message template)
    category: str = "technical"
    severity: float = 0.5
    triggers: tuple = ()


FAMILIES = (
    Family("safety", (
        ("NH", "Not-Halt Kreis {n} betaetigt"),
        ("LV", "Lichtvorhang {n} unterbrochen"),
        ("SR", "Sicherheitsrelais {n} hat abgeschaltet"),
        ("TV", "Zugangstor {n} nicht verriegelt"),
        ("ZH", "Zweihandbedienung {n} gestoert"),
    ), severity=0.95, triggers=("drive", "robot", "clamp", "conveyor")),
    Family("pneumatic", (
        ("VE", "Ventil {n} ohne Rueckmeldung"),
        ("ZY", "Zylinder {n} Endlage vorn nicht erreicht"),
        ("ZY", "Zylinder {n} Endlage hinten nicht erreicht"),
        ("DR", "Druckluft {n} unter Mindestdruck"),
        ("WE", "Wartungseinheit {n} Filter zugesetzt"),
    ), severity=0.55, triggers=("clamp", "quality")),
    Family("drive", (
        ("SA", "Achse {n} Schleppabstand zu gross"),
        ("FU", "Umrichter {n} Ueberstrom"),
        ("MO", "Motor {n} Uebertemperatur"),
        ("SA", "Achse {n} nicht referenziert"),
        ("BR", "Haltebremse Achse {n} oeffnet nicht"),
    ), severity=0.75, triggers=("robot", "conveyor", "quality")),
    Family("robot", (
        ("RB", "Roboter {n} nicht in Grundstellung"),
        ("WW", "Werkzeugwechsler {n} ohne Quittung"),
        ("RB", "Roboter {n} Bahnabweichung"),
        ("RB", "Roboter {n} Kollisionserkennung aktiv"),
        ("GR", "Greifer {n} meldet kein Teil"),
    ), severity=0.8, triggers=("quality", "clamp")),
    Family("conveyor", (
        ("FB", "Foerderband {n} Stau erkannt"),
        ("HT", "Hubtisch {n} Position oben fehlt"),
        ("WT", "Werkstuecktraeger {n} nicht gelesen"),
        ("FB", "Transportfreigabe {n} fehlt"),
        ("FB", "Foerderband {n} Sensor dauernd belegt"),
    ), severity=0.4, triggers=("buffer", "quality")),
    Family("clamp", (
        ("SPN", "Spanner {n} Endlage zu fehlt"),
        ("SPN", "Spanner {n} Endlage auf fehlt"),
        ("VR", "Verriegelung {n} nicht in Position"),
        ("NP", "Nullpunktspanner {n} nicht gespannt"),
        ("AL", "Bauteil {n} liegt nicht auf"),
    ), severity=0.6, triggers=("quality", "robot")),
    Family("sensor", (
        ("LS", "Lichtschranke {n} verschmutzt"),
        ("IS", "Initiator {n} kein Signal"),
        ("KA", "Kamera {n} Auswertung nicht moeglich"),
        ("MF", "Messfuehler {n} Wert ausserhalb Bereich"),
        ("SE", "Sensor {n} Abgleich erforderlich"),
    ), severity=0.45, triggers=("quality",)),
    Family("media", (
        ("KW", "Kuehlwasser {n} Durchfluss gering"),
        ("AB", "Absaugung {n} Unterdruck fehlt"),
        ("HY", "Hydraulik {n} Druck zu niedrig"),
        ("SM", "Schmierung {n} Intervall abgelaufen"),
        ("LK", "Leckage {n} erkannt"),
    ), severity=0.65, triggers=("drive", "quality")),
    Family("quality", (
        ("QS", "Pruefung {n} nicht in Ordnung"),
        ("TL", "Toleranz {n} ueberschritten"),
        ("NA", "Nacharbeit {n} angefordert"),
        ("ID", "Bauteilkennung {n} nicht lesbar"),
        ("MW", "Messwert {n} unplausibel"),
    ), severity=0.5),
    Family("buffer", (
        ("PU", "Puffer {n} voll"),
        ("PU", "Puffer {n} leer"),
        ("LA", "Auslagern {n} blockiert"),
        ("LA", "Einlagern {n} nicht freigegeben"),
    ), severity=0.35, triggers=("conveyor",)),
    Family("organization", (
        ("MA", "Material {n} fehlt am Platz"),
        ("OR", "Schichtwechsel Bereich {n}"),
        ("OR", "Wartung Station {n} geplant"),
        ("OR", "Werker {n} Eingriff angefordert"),
        ("OR", "Typwechsel {n} laeuft"),
    ), category="organizational", severity=0.3),
    Family("misc", (
        ("NW", "Teilnehmer {n} Kommunikation gestoert"),
        ("SY", "Uhrzeit {n} nicht synchron"),
        ("SY", "Diagnosespeicher {n} voll"),
        ("SY", "Meldung {n} ohne Text"),
    ), category="other", severity=0.25),
)

FAMILY = {f.key: f for f in FAMILIES}

ROLE_FAMILIES = {
    "PRS": {"drive": 0.3, "pneumatic": 0.25, "safety": 0.1, "media": 0.15, "quality": 0.1, "organization": 0.08, "misc": 0.02},
    "SWE": {"robot": 0.3, "clamp": 0.22, "drive": 0.15, "safety": 0.12, "media": 0.09, "quality": 0.07, "organization": 0.04, "misc": 0.01},
    "ROB": {"robot": 0.4, "drive": 0.18, "safety": 0.12, "clamp": 0.12, "sensor": 0.09, "quality": 0.05, "organization": 0.03, "misc": 0.01},
    "KLB": {"pneumatic": 0.3, "media": 0.22, "robot": 0.16, "quality": 0.14, "sensor": 0.1, "organization": 0.06, "misc": 0.02},
    "FOE": {"conveyor": 0.45, "sensor": 0.18, "drive": 0.14, "safety": 0.1, "buffer": 0.07, "organization": 0.05, "misc": 0.01},
    "PUF": {"buffer": 0.42, "conveyor": 0.25, "sensor": 0.13, "drive": 0.1, "organization": 0.08, "misc": 0.02},
    "FTS": {"conveyor": 0.38, "sensor": 0.22, "drive": 0.16, "safety": 0.13, "organization": 0.09, "misc": 0.02},
    "MON": {"pneumatic": 0.24, "clamp": 0.2, "robot": 0.16, "quality": 0.14, "sensor": 0.12, "safety": 0.08, "organization": 0.05, "misc": 0.01},
    "MES": {"sensor": 0.35, "quality": 0.3, "drive": 0.12, "conveyor": 0.1, "organization": 0.1, "misc": 0.03},
    "PRF": {"quality": 0.38, "sensor": 0.22, "media": 0.14, "safety": 0.1, "drive": 0.08, "organization": 0.06, "misc": 0.02},
}
