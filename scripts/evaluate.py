#!/usr/bin/env python3
"""
Evaluiere trainiertes YOLOv8 Modell auf Test-Datensatz
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Fehler: ultralytics nicht installiert.")
    sys.exit(1)

from config import DATA_DIR, MODEL_DIR


def evaluate_model(model_path, data_yaml_path):
    """
    Evaluiert das Modell
    
    Args:
        model_path: Pfad zum trainierten Modell
        data_yaml_path: Pfad zur data.yaml
    
    Returns:
        Evaluierungs-Metriken
    """
    
    # Überprüfe Dateien
    if not os.path.exists(model_path):
        print(f"❌ Fehler: Modell nicht gefunden: {model_path}")
        return None
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ Fehler: data.yaml nicht gefunden: {data_yaml_path}")
        return None
    
    print("📊 Starte Evaluierung...")
    print(f"   Modell: {model_path}")
    print(f"   Datensatz: {data_yaml_path}")
    
    model = YOLO(model_path)
    
    # Validiere
    metrics = model.val(data=data_yaml_path)
    
    print("\n✅ Evaluierung abgeschlossen!")
    print("\n📈 Metriken:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    
    return metrics


def test_single_image(model_path, image_path, conf_threshold=0.5):
    """
    Testet das Modell auf einem Bild
    
    Args:
        model_path: Pfad zum trainierten Modell
        image_path: Pfad zum Test-Bild
        conf_threshold: Konfidenz-Schwellwert
    
    Returns:
        Vorhersage-Ergebnisse
    """
    
    if not os.path.exists(image_path):
        print(f"❌ Fehler: Bild nicht gefunden: {image_path}")
        return None
    
    print(f"🎯 Teste Bild: {image_path}")
    
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold)
    
    if results:
        result = results[0]
        print(f"\n✅ Erkannte Objekte:")
        
        if result.boxes:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                print(f"   - {class_name}: {confidence:.2%}")
        else:
            print("   Keine Objekte erkannt")
    
    return results


if __name__ == "__main__":
    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    best_model = os.path.join(MODEL_DIR, "tennis_detector", "weights", "best.pt")
    
    # Evaluiere Modell
    metrics = evaluate_model(best_model, data_yaml)
