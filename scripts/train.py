#!/usr/bin/env python3
"""
YOLOv8 Modell für Tennis-Objekterkennung trainieren
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Fehler: ultralytics nicht installiert. Bitte 'pip install ultralytics' ausführen.")
    sys.exit(1)

from config import MODEL_SIZE, TRAINING_CONFIG, DATA_DIR, MODEL_DIR, RESULTS_DIR


def train_model(data_yaml_path, model_size=MODEL_SIZE, config=TRAINING_CONFIG, task="detect"):
    """
    Trainiert YOLOv8 Modell
    
    Args:
        data_yaml_path: Pfad zur data.yaml
        model_size: Modellgröße (n, s, m, l, x)
        config: Training-Konfiguration
        task: Modelltyp ("detect" oder "pose")
    
    Returns:
        Trainings-Ergebnisse
    """
    
    # Überprüfe data.yaml
    if not os.path.exists(data_yaml_path):
        print(f"❌ Fehler: {data_yaml_path} nicht gefunden!")
        print("Bitte erst Datensatz mit 'python scripts/download_dataset.py' herunterladen.")
        return None
    
    # Erstelle Model-Verzeichnis
    Path(MODEL_DIR).mkdir(exist_ok=True)
    
    print("🚀 Starte YOLOv8 Training...")
    print(f"   Modellgröße: {model_size}")
    print(f"   Task: {task}")
    print(f"   Epochen: {config['epochs']}")
    print(f"   Batch-Größe: {config['batch_size']}")
    print(f"   Bilder-Größe: {config['imgsz']}")
    
    # Lade vortrainiertes Modell
    model_suffix = "-pose" if task == "pose" else ""
    model_name = f"yolov8{model_size}{model_suffix}.pt"
    print(f"   Modell: {model_name}")
    
    model = YOLO(model_name)
    
    # Trainiere
    results = model.train(
        data=data_yaml_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch_size"],
        device=config["device"],
        patience=config["patience"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        augment=config["augment"],
        project=MODEL_DIR,
        name="tennis_detector",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )
    
    print("✅ Training abgeschlossen!")
    print(f"   Beste Gewichte: {MODEL_DIR}/tennis_detector/weights/best.pt")
    
    return results


def get_best_model():
    """
    Gibt Pfad zum besten trainierten Modell zurück
    """
    best_model = os.path.join(MODEL_DIR, "tennis_detector", "weights", "best.pt")
    
    if os.path.exists(best_model):
        return best_model
    else:
        print(f"❌ Bestes Modell nicht gefunden: {best_model}")
        return None


if __name__ == "__main__":
    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    
    # Trainiere Modell
    results = train_model(data_yaml)
    
    if results:
        best_model = get_best_model()
        if best_model:
            print(f"\n💾 Modell gespeichert: {best_model}")
