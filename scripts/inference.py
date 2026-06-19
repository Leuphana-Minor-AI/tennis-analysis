#!/usr/bin/env python3
"""
Verwende trainiertes Modell für Vorhersagen
"""

import os
import sys
import cv2
from pathlib import Path

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

try:
    from ultralytics import YOLO
    import numpy as np
    import torch
except ImportError:
    print("Fehler: Erforderliche Pakete nicht installiert.")
    sys.exit(1)

# Patch torch.load to handle PyTorch 2.6+ security restrictions
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Wrapper for torch.load that disables weights_only for model loading"""
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

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


def predict_frames(model_path, frames_dir="data/frames", max_frames=10, conf_threshold=0.5, save_dir="results/predictions"):
    """
    Vorhersage auf mehrere Bilder aus einem Verzeichnis
    
    Args:
        model_path: Pfad zum trainierten Modell
        frames_dir: Verzeichnis mit Bildern
        max_frames: Maximale Anzahl der Bilder zum Testen
        conf_threshold: Konfidenz-Schwellwert
        save_dir: Verzeichnis zum Speichern der Ergebnisse
    """
    
    if not os.path.exists(frames_dir):
        print(f"ERROR: Directory not found: {frames_dir}")
        return None
    
    print(f"Testing model on frames from: {frames_dir}")
    
    # Finde alle JPG-Dateien
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print(f"ERROR: No images found in {frames_dir}")
        return None
    
    print(f"   Found: {len(frame_files)} images")
    print(f"   Testing: {min(max_frames, len(frame_files))} images\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model = YOLO(model_path)
    
    total_detections = 0
    tested_frames = 0
    
    for frame_file in frame_files[:max_frames]:
        frame_path = os.path.join(frames_dir, frame_file)
        results = model.predict(frame_path, conf=conf_threshold, verbose=False)
        
        result = results[0]
        num_detections = len(result.boxes)
        total_detections += num_detections
        tested_frames += 1
        
        # Save annotated image
        annotated_image = result.plot()  # This returns RGB image with boxes drawn
        save_path = os.path.join(save_dir, f"annotated_{frame_file}")
        cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        if num_detections > 0:
            print(f"   {frame_file}: {num_detections} objects detected")
        
        if tested_frames % 5 == 0:
            print(f"   ... {tested_frames}/{min(max_frames, len(frame_files))} images processed")
    
    print(f"\nTest completed!")
    print(f"   Images tested: {tested_frames}")
    print(f"   Total objects detected: {total_detections}")
    if tested_frames > 0:
        print(f"   Average: {total_detections/tested_frames:.1f} objects per image")
    print(f"   Results saved to: {save_dir}")
    
    return total_detections


if __name__ == "__main__":
    # Suche nach dem Modell
    model_candidates = [
        os.path.join(MODEL_DIR, "yolo26s-pose.pt"),  # Hochgeladenes Modell
        os.path.join(MODEL_DIR, "tennis_detector", "weights", "best.pt"),  # Trainiertes Modell
    ]
    
    best_model = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            best_model = candidate
            break
    
    if best_model is None:
        print(f"ERROR: No model found")
        print(f"Versucht: {model_candidates}")
        sys.exit(1)
    
    print(f"Using model: {best_model}\n")
    
    # Beispiel: Vorhersage auf Bild oder teste auf Frames-Verzeichnis
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(best_model, image_path)
    else:
        # Automatisch auf Frames-Verzeichnis testen
        frames_dir = "data/frames"
        if os.path.exists(frames_dir):
            predict_frames(best_model, frames_dir, max_frames=20)
        else:
            print("Usage:")
            print("  python scripts/inference.py <image_path>")
            print("\nExample:")
            print("  python scripts/inference.py data/frames/frame_0000.jpg")
    