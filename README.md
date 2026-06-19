# 🎾 Tennis Analysis - YOLOv8 Objektdetektor

Ein professionelles System zur Erkennung von **Tennisbällen**, **Schlägern**, **Spielern** und anderen Tennis-Objekten mit YOLOv8 und Roboflow.

---

## ⚡ Schnellstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Konfiguration (config.py)

```python
ROBOFLOW_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "workspace": "your_workspace",
    "project": "tennis_detection",
    "version": 1,
}

MODEL_SIZE = "m"  # n (nano), s (small), m (medium), l (large), x (xlarge)

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
}
```

### 3. Datensatz herunterladen

```bash
python scripts/download_dataset.py
```

### 4. Modell trainieren

```bash
python scripts/train.py
```

### 5. Modell evaluieren

```bash
python scripts/evaluate.py
```

### 6. Vorhersagen machen

```bash
# Auf Bild
python scripts/inference.py path/to/image.jpg

# Auf Video
python scripts/inference.py path/to/video.mp4

# Live von Webcam
python -c "from scripts.inference import predict_webcam; predict_webcam('models/tennis_detector/weights/best.pt')"
```

---

## 📁 Projektstruktur

```
tennis-analysis/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   └── data.yaml          # Roboflow generierte Config
├── models/
│   └── tennis_detector/
│       └── weights/
│           ├── best.pt    # Bestes Modell
│           └── last.pt    # Letztes Modell
├── results/               # Vorhersage-Ergebnisse
├── scripts/
│   ├── download_dataset.py    # Roboflow Download
│   ├── train.py               # Training
│   ├── evaluate.py            # Evaluierung
│   └── inference.py           # Vorhersagen
├── config.py              # Zentrale Konfiguration
├── requirements.txt       # Python Dependencies
└── README.md
```

---

## 🔧 Detaillierte Anleitung

### Schritt 1: Roboflow Setup

1. Registriere auf [Roboflow.com](https://roboflow.com/)
2. Erstelle neues Projekt
3. Lade Bilder hoch oder verwende öffentliche Datensätze
4. **Annotiere** die Objekte (Ball, Schläger, Spieler)
5. Generiere Datensatz im **YOLOv8 Format**
6. Kopiere API Key aus Account Settings

### Schritt 2: Training starten

```bash
# Mit standard Einstellungen
python scripts/train.py

# Training wird gestartet...
# - Modell: yolov8m.pt (Medium - guter Trade-off)
# - Epochen: 100
# - Batch: 16
# - Größe: 640x640

# Nach ~2-4 Stunden (je nach GPU) fertig
```

### Schritt 3: Ergebnisse prüfen

```bash
python scripts/evaluate.py
# Zeigt mAP, Precision, Recall, etc.
```

### Schritt 4: In Produktion

```python
from ultralytics import YOLO
import cv2

# Modell laden
model = YOLO("models/tennis_detector/weights/best.pt")

# Bild/Video verarbeiten
results = model.predict("test.jpg", conf=0.5)

# Ergebnisse anzeigen
for result in results:
    print(result.boxes)  # Erkannte Boxen
```

---

## 📊 Modellgrößen

| Größe | Parameter | Speed (ms) | mAP50 | Nutzung |
|-------|-----------|-----------|-------|---------|
| **n** | 2.6M | 1 | ~30 | Edge Devices, Mobile |
| **s** | 9.2M | 12 | ~40 | Schnelle Erkennung |
| **m** | 25M | 25 | ~50 | **Empfohlen** |
| **l** | 52M | 40 | ~55 | High Accuracy |
| **x** | 107M | 60 | ~57 | Maximum Accuracy |

---

## 🎯 Best Practices

### Datensatz

- Mindestens **100-200 Bilder** pro Klasse
- **Verschiedene Winkel** und Lichtverhältnisse
- **Augmentation** in Roboflow aktivieren (Flip, Rotation, Blur)
- Train/Val/Test: **70/20/10** Verhältnis

### Training

```python
TRAINING_CONFIG = {
    "epochs": 100,           # Mehr Epochen = bessere Genauigkeit
    "batch_size": 16,        # Höher = schneller, aber mehr RAM
    "imgsz": 640,            # Höher = besser für kleine Objekte
    "patience": 20,          # Early Stopping
    "optimizer": "auto",     # Automatisch optimieren
}
```

### Evaluierung

```bash
python scripts/evaluate.py
# mAP50 > 0.5 = gut
# mAP50 > 0.7 = sehr gut
# mAP50 > 0.85 = ausgezeichnet
```

---

## 🐛 Häufige Probleme

### "ModuleNotFoundError: No module named 'roboflow'"
```bash
pip install roboflow
```

### GPU nicht erkannt
```bash
# CUDA Check
python -c "import torch; print(torch.cuda.is_available())"

# CPU Fallback in config.py:
# "device": -1  # CPU
```

### Zu wenige Erkennungen
- Datensatz vergrößern
- Model Size erhöhen (n → m → l)
- Confidence Threshold senken

---

## 📚 Ressourcen

- 🎓 [YOLOv8 Dokumentation](https://docs.ultralytics.com/)
- 📊 [Roboflow Dokumentation](https://docs.roboflow.com/)
- 🎬 [YOLOv8 Tutorial Videos](https://www.youtube.com/@ultralytics)
- 💻 [GitHub Ultralytics](https://github.com/ultralytics/ultralytics)
