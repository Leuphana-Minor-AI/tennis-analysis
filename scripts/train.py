#!/usr/bin/env python3
"""
YOLOv8 Modell für Tennis-Objekterkennung trainieren
Unterstützt beide Detection und Pose Estimation Modelle
"""

import os
import sys
import argparse
from pathlib import Path

# Füge parent directory zu Python path hinzu um config zu importieren
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Fehler: ultralytics nicht installiert. Bitte 'pip install ultralytics' ausführen.")
    sys.exit(1)

from config import MODEL_SIZE, TRAINING_CONFIG, DATA_DIR, MODEL_DIR, RESULTS_DIR


def train_model(data_yaml_path, model_name="yolov8x.pt", config=TRAINING_CONFIG, task="detect"):
    """
    Trainiert YOLOv8 Modell (Detection oder Pose)
    
    Args:
        data_yaml_path: Pfad zur data.yaml
        model_name: Modellname (z.B. "yolov8x.pt" oder "yolo26s-pose.pt")
        config: Training-Konfiguration
        task: Aufgabe ("detect" oder "pose")
    
    Returns:
        Trainings-Ergebnisse
    """
    
    # Überprüfe data.yaml
    if not os.path.exists(data_yaml_path):
        print(f"❌ Fehler: {data_yaml_path} nicht gefunden!")
        print("Bitte erst das Dataset konvertieren: python scripts/convert_ndjson_to_yolo.py")
        return None
    
    # Erstelle Model-Verzeichnis
    Path(MODEL_DIR).mkdir(exist_ok=True)
    
    print("🚀 Starte YOLO Training...")
    print(f"   Task: {task}")
    print(f"   Modell: {model_name}")
    print(f"   Data: {data_yaml_path}")
    print(f"   Epochen: {config['epochs']}")
    print(f"   Batch-Größe: {config['batch_size']}")
    print(f"   Bilder-Größe: {config['imgsz']}")
    
    # Lade vortrainiertes Modell
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
        name=f"tennis_{task}",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )
    
    print("✅ Training abgeschlossen!")
    print(f"   Beste Gewichte: {MODEL_DIR}/tennis_{task}/weights/best.pt")
    
    return results


def get_best_model(task="detect"):
    """
    Gibt Pfad zum besten trainierten Modell zurück
    """
    best_model = os.path.join(MODEL_DIR, f"tennis_{task}", "weights", "best.pt")
    
    if os.path.exists(best_model):
        return best_model
    else:
        print(f"❌ Bestes Modell nicht gefunden: {best_model}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainiere YOLO Modell für Tennis-Erkennung")
    parser.add_argument("--data", type=str, default="data/tennis_keypoints/data.yaml",
                        help="Pfad zur data.yaml Datei (default: data/tennis_keypoints/data.yaml)")
    parser.add_argument("--model", type=str, default="yolo26s-pose.pt",
                        help="Modellname (default: yolo26s-pose.pt)")
    parser.add_argument("--task", type=str, choices=["detect", "pose"], default="pose",
                        help="Training Task (default: pose)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Anzahl Epochen (überschreibt config)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch Size (überschreibt config)")
    
    args = parser.parse_args()
    
    # Nutze config, aber erlaube Kommandozeilenüberschreibungen
    config = TRAINING_CONFIG.copy()
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch_size"] = args.batch
    
    # Überprüfe ob Modell-Datei existiert
    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.exists(model_path) and not os.path.exists(args.model):
        print(f"⚠️ Warnung: Modell {args.model} nicht gefunden, wird von Hugging Face heruntergeladen...")
    
    # Trainiere Modell
    results = train_model(args.data, args.model, config, args.task)
    
    if results:
        best_model = get_best_model(args.task)
        if best_model:
            print(f"\n🎯 Best model: {best_model}")
        if best_model:
            print(f"\n💾 Modell gespeichert: {best_model}")
