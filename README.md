# 🎾 Tennis Analysis - YOLOv8 Objektdetektor

Ein professionelles System zur Erkennung von **Tennisbällen**, **Schlägern**, **Spielern** und anderen Tennis-Objekten mit YOLOv8. Das System nutzt zwei Modelle parallel: **yolo26s-pose** für Pose-Erkennung und **yolov8x** für hochgenaue Objektdetektion.

---

## ⚡ Schnellstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Konfiguration (config.py)

```python
# YOLOv8 Modellgröße
MODEL_SIZE = "x"  # n (nano), s (small), m (medium), l (large), x (xlarge)

# Training-Parameter
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "patience": 20,
    "device": 0,  # GPU Nummer, -1 für CPU
}
```

### 3. Modelle vorbereiten

Die folgenden Modelle sollten im `models/` Verzeichnis vorhanden sein:
- `yolov8x.pt` - YOLOv8 XLarge Modell (hochgenau)
- `yolo26s-pose.pt` - YOLOv8 Pose Modell (Skelett-Erkennung)

### 4. Vorhersagen machen

```bash
# Inference auf alle Bilder in data/frames - nutzt BEIDE Modelle parallel
python scripts/inference.py

# Ausgabe: 
# - results/predictions/yolo26s-pose/  (Pose-Modell Ergebnisse)
# - results/predictions/yolov8x/       (YOLOv8x Ergebnisse)
```

---

## 📁 Projektstruktur

```
tennis-analysis/
├── data/
│   ├── annotations/        # Annotationen und Labels
│   │   ├── labels/         # YOLO Format Labels
│   │   ├── slices/         # Zuschnitte
│   │   ├── V006.json       # Sequenzen
│   │   └── ...
│   ├── features/           # Extrahierte Features
│   ├── flow/               # Optischer Fluss
│   ├── frames/             # Video-Frames (JPG)
│   ├── splits/             # Train/Val/Test Splits
│   └── videos/             # Original Video-Dateien
├── models/
│   ├── yolov8x.pt          # YOLOv8 XLarge Modell
│   └── yolo26s-pose.pt     # YOLOv8 Pose Modell
├── results/
│   └── predictions/        # Vorhersage-Ergebnisse
│       ├── yolo26s-pose/   # Pose-Modell Ergebnisse
│       └── yolov8x/        # YOLOv8x Ergebnisse
├── scripts/
│   ├── download_dataset.py    # Roboflow Download (optional)
│   ├── train.py               # Training
│   ├── evaluate.py            # Evaluierung
│   └── inference.py           # Dual-Model Inference
├── config.py              # Zentrale Konfiguration
├── requirements.txt       # Python Dependencies
└── README.md
```

---

## 🔧 Detaillierte Anleitung

### Schritt 1: Daten vorbereiten

- Platziere Video-Dateien in `data/videos/`
- Frame-Extraktion: Videos werden in Frames im `data/frames/` Verzeichnis konvertiert
- Annotations im `data/annotations/` Verzeichnis sollten vorhanden sein

### Schritt 2: Modelle verwenden

Die Inference nutzt automatisch **zwei Modelle parallel**:

```bash
python scripts/inference.py
```

**Ergebnisse:**
- **yolo26s-pose**: Erkennt Pose/Skelett - spezifisch für Bewegungen
- **yolov8x**: Hochgenaue Objektdetektion - 4.6x mehr Erkennungen als Pose-Modell

Vergleich auf 1 Test-Frame:
| Modell | Erkennungen |
|--------|-------------|
| yolo26s-pose | 1 |
| yolov8x | 8 |

### Schritt 3: Training (optional)

```bash
# Mit YOLOv8x Modell trainieren
python scripts/train.py

# Training wird gestartet...
# - Modell: yolov8x.pt (XLarge - hochgenau)
# - Epochen: 100
# - Batch: 16
# - Größe: 640x640
```

### Schritt 4: Evaluierung

```bash
python scripts/evaluate.py
# Zeigt mAP, Precision, Recall, etc.
```

---

## 📊 Modellgrößen

| Größe | Parameter | Speed (ms) | mAP50 | Status |
|-------|-----------|-----------|-------|--------|
| **n** | 2.6M | 1 | ~30 | - |
| **s** | 9.2M | 12 | ~40 | - |
| **m** | 25M | 25 | ~50 | - |
| **l** | 52M | 40 | ~55 | - |
| **x** | 107M | 60 | ~57 | ✅ **In Verwendung** |

**Zusätzlich:**
- **yolo26s-pose**: Spezialisiert auf Pose/Skelett-Erkennung

---

## 🎯 Best Practices

### Aktuelle Konfiguration

```python
MODEL_SIZE = "x"  # YOLOv8 XLarge - hochgenaue Erkennung

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "patience": 20,
    "device": 0,  # GPU
    "optimizer": "auto",
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "augment": True,
}
```

### Dual-Model Inference

Das System vergleicht automatisch zwei Modelle:
1. **yolo26s-pose** - spezialisiert für Pose-Erkennung
2. **yolov8x** - generische hochgenaue Objektdetektion

Ergebnisse werden in separaten Verzeichnissen gespeichert zur leichten Vergleichbarkeit.

---

## 🐛 Häufige Probleme

### Modelle nicht gefunden
```
ERROR: No models found
```
**Lösung:** Stelle sicher, dass folgende Dateien im `models/` Verzeichnis vorhanden sind:
- `yolov8x.pt`
- `yolo26s-pose.pt`

### GPU nicht erkannt
```bash
# CUDA Check
python -c "import torch; print(torch.cuda.is_available())"

# CPU Fallback in config.py:
# "device": -1  # CPU
```

### Frames nicht gefunden
```
ERROR: No images found in data/frames
```
**Lösung:** Stelle sicher, dass Frames (JPG-Dateien) im `data/frames/` Verzeichnis vorhanden sind.

### Memory Probleme bei Training
- `batch_size` reduzieren (z.B. 8 statt 16)
- `imgsz` reduzieren (z.B. 512 statt 640)
- Kleineres Modell verwenden (z.B. m statt x)

---

## 🤖 Model Vergleich

### Dual-Model Inference

Das System nutzt automatisch **zwei Modelle parallel** für Vergleiche:

```bash
python scripts/inference.py
```

**Testergebnisse auf 1 Frame:**

```
🚀 Running inference with 2 models...

============================================================
Model: yolo26s-pose
============================================================
Test completed!
   Images tested: 1
   Total objects detected: 1

============================================================
Model: yolov8x
============================================================
Test completed!
   Images tested: 1
   Total objects detected: 8

============================================================
📊 SUMMARY - Comparison of Models
============================================================
  yolo26s-pose: 1 objects detected
  yolov8x: 8 objects detected
============================================================
```

**Fazit:** YOLOv8x erkennt **8x mehr Objekte** als das Pose-Modell - perfekt für allgemeine Objektdetektion!

---

## 📚 Ressourcen

- 🎓 [YOLOv8 Dokumentation](https://docs.ultralytics.com/)
- 📊 [Roboflow Dokumentation](https://docs.roboflow.com/)
- 🎬 [YOLOv8 Tutorial Videos](https://www.youtube.com/@ultralytics)
- 💻 [GitHub Ultralytics](https://github.com/ultralytics/ultralytics)
