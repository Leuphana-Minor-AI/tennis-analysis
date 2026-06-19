#!/usr/bin/env python3
"""
Verwende trainiertes Modell für Vorhersagen
"""

import os
import sys
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
    import numpy as np
except ImportError:
    print("Fehler: Erforderliche Pakete nicht installiert.")
    sys.exit(1)

from config import MODEL_DIR


def predict_image(model_path, image_path, conf_threshold=0.5, save_result=True):
    """
    Vorhersage auf Bild
    
    Args:
        model_path: Pfad zum trainierten Modell
        image_path: Pfad zum Bild
        conf_threshold: Konfidenz-Schwellwert
        save_result: Speichere Ergebnis-Bild
    
    Returns:
        Ergebnisse
    """
    
    if not os.path.exists(image_path):
        print(f"❌ Fehler: Bild nicht gefunden: {image_path}")
        return None
    
    print(f"🎯 Vorhersage auf Bild: {image_path}")
    
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold, save=save_result)
    
    result = results[0]
    
    print(f"\n✅ Erkannte Objekte:")
    
    if result.boxes:
        for i, box in enumerate(result.boxes):
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0]
            print(f"   {i+1}. {class_name}: {confidence:.2%} (x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f})")
    else:
        print("   Keine Objekte erkannt")
    
    return results


def predict_video(model_path, video_path, conf_threshold=0.5, output_path="output.mp4"):
    """
    Vorhersage auf Video
    
    Args:
        model_path: Pfad zum trainierten Modell
        video_path: Pfad zum Video
        conf_threshold: Konfidenz-Schwellwert
        output_path: Ausgabepfad für verarbeitetes Video
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Fehler: Video nicht gefunden: {video_path}")
        return None
    
    print(f"🎬 Verarbeite Video: {video_path}")
    
    model = YOLO(model_path)
    
    # Vorhersage auf Video
    results = model.predict(
        video_path,
        conf=conf_threshold,
        save=True,
        project="results",
        name="predictions",
        exist_ok=True,
    )
    
    print(f"✅ Video verarbeitet!")
    print(f"   Ausgabe: results/predictions/")
    
    return results


def predict_webcam(model_path, conf_threshold=0.5, exit_key="q"):
    """
    Live-Vorhersage von Webcam
    
    Args:
        model_path: Pfad zum trainierten Modell
        conf_threshold: Konfidenz-Schwellwert
        exit_key: Taste zum Beenden (Standard: 'q')
    """
    
    print(f"📹 Starte Webcam (Drücke '{exit_key}' zum Beenden)...")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Fehler: Webcam nicht verfügbar")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Vorhersage
        results = model(frame, conf=conf_threshold)
        
        # Zeichne Boxen
        annotated_frame = results[0].plot()
        
        # Zeige Bild
        cv2.imshow("Tennis Detektor", annotated_frame)
        
        # Exit
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam beendet")


if __name__ == "__main__":
    best_model = os.path.join(MODEL_DIR, "tennis_detector", "weights", "best.pt")
    
    if not os.path.exists(best_model):
        print(f"❌ Fehler: Modell nicht gefunden: {best_model}")
        print("Bitte erst mit 'python scripts/train.py' trainieren.")
        sys.exit(1)
    
    # Beispiel: Vorhersage auf Bild
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(best_model, image_path)
    else:
        print("Verwendung:")
        print("  python scripts/inference.py <image_path>")
        print("\nBeispiel:")
        print("  python scripts/inference.py test.jpg")
