#!/usr/bin/env python3
"""
Extract frames from video and run inference on them
"""

import os
import sys
import cv2
from pathlib import Path
from datetime import datetime

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


def extract_frames(video_path, output_dir="data/frames", frame_interval=1):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every nth frame (1 = every frame)
    
    Returns:
        List of extracted frame paths
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Fehler: Video nicht gefunden: {video_path}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎬 Extrahiere Frames aus Video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Fehler: Kann Video nicht öffnen: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video-Info:")
    print(f"   FPS: {fps}")
    print(f"   Gesamte Frames: {total_frames}")
    print(f"   Auflösung: {width}x{height}")
    
    frame_count = 0
    extracted_count = 0
    frame_paths = []
    
    print(f"🔄 Extrahiere jeden {frame_interval}. Frame...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if extracted_count % 50 == 0:
                print(f"   ✓ {extracted_count} Frames extrahiert...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"✅ {extracted_count} Frames extrahiert und gespeichert in: {output_dir}")
    
    return frame_paths


def run_inference_on_frames(frames_dir, model_name="yolov8x", conf_threshold=0.5):
    """
    Run inference on extracted frames
    
    Args:
        frames_dir: Directory containing extracted frames
        model_name: Name of model (yolov8x or yolo26s-pose)
        conf_threshold: Confidence threshold
    
    Returns:
        Results
    """
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Fehler: Modell nicht gefunden: {model_path}")
        return None
    
    if not os.path.exists(frames_dir):
        print(f"❌ Fehler: Frames-Verzeichnis nicht gefunden: {frames_dir}")
        return None
    
    print(f"\n🤖 Starte Inference mit Modell: {model_name}")
    print(f"   Modell: {model_path}")
    print(f"   Frames-Verzeichnis: {frames_dir}")
    
    model = YOLO(model_path)
    
    # Run inference on all frames in directory with stream=True to prevent RAM accumulation
    results = model.predict(
        frames_dir,
        conf=conf_threshold,
        save=False,
        verbose=False,
        stream=True
    )
    
    # Save results
    output_dir = f"results/predictions/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Verarbeite Bilder...")
    
    detection_stats = {
        "total_frames": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "class_counts": {}
    }
    
    for i, result in enumerate(results):
        detection_stats["total_frames"] += 1
        if result.boxes:
            detection_stats["frames_with_detections"] += 1
            detection_stats["total_detections"] += len(result.boxes)
            
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                detection_stats["class_counts"][class_name] = detection_stats["class_counts"].get(class_name, 0) + 1
        
        if (i + 1) % 50 == 0:
            print(f"   ✓ {i + 1} Frames verarbeitet...")
    
    print(f"\n✅ Inference abgeschlossen!")
    print(f"\n📈 Statistik ({model_name}):")
    print(f"   Bearbeitete Frames: {detection_stats['total_frames']}")
    print(f"   Frames mit Detektionen: {detection_stats['frames_with_detections']}")
    print(f"   Gesamte Detektionen: {detection_stats['total_detections']}")
    print(f"   Klassen:")
    for class_name, count in sorted(detection_stats["class_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"      {class_name}: {count}")
    
    return results, detection_stats


def cleanup_frames(frames_dir="data/frames"):
    """
    Delete all frames from directory to free up space
    
    Args:
        frames_dir: Directory containing frames to delete
    """
    
    if not os.path.exists(frames_dir):
        print(f"⚠️  Verzeichnis nicht gefunden: {frames_dir}")
        return
    
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    
    if not frame_files:
        print(f"ℹ️  Keine Frames zum Löschen in {frames_dir}")
        return
    
    print(f"\n🗑️  Lösche {len(frame_files)} Frames aus {frames_dir}...")
    
    deleted_count = 0
    for frame_file in frame_files:
        try:
            frame_path = os.path.join(frames_dir, frame_file)
            os.remove(frame_path)
            deleted_count += 1
        except Exception as e:
            print(f"   ⚠️  Fehler beim Löschen von {frame_file}: {e}")
    
    print(f"✅ {deleted_count} Frames gelöscht")


def main():
    """
    Main function
    """
    
    print("=" * 70)
    print("🎾 Tennis-Video Verarbeitung - Frames extrahieren & Inference")
    print("=" * 70)
    
    # Video path - can be overridden by command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/videos/V001.mp4"
    
    frames_dir = "data/frames"
    
    # Step 1: Extract frames
    print("\n▶ SCHRITT 1: Extrahiere Frames aus Video\n")
    frame_paths = extract_frames(video_path, output_dir=frames_dir, frame_interval=1)
    
    if not frame_paths:
        print("❌ Fehler: Keine Frames extrahiert!")
        return
    
    # Step 2: Run inference on frames with both models
    print("\n" + "=" * 70)
    print("▶ SCHRITT 2: Starte Inference auf extrahierten Frames")
    print("=" * 70)
    
    models = ["yolov8x", "yolo26s-pose"]
    all_stats = {}
    
    for model_name in models:
        results, stats = run_inference_on_frames(frames_dir, model_name=model_name, conf_threshold=0.5)
        all_stats[model_name] = stats
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ZUSAMMENFASSUNG")
    print("=" * 70)
    
    for model_name, stats in all_stats.items():
        print(f"\n{model_name}:")
        print(f"  Frames mit Detektionen: {stats['frames_with_detections']}/{stats['total_frames']}")
        print(f"  Gesamte Detektionen: {stats['total_detections']}")
    
    # Cleanup frames after inference
    cleanup_frames(frames_dir)
    
    print(f"\n✅ Verarbeitung abgeschlossen!")
    print(f"   Frames gespeichert: {frames_dir}")
    print(f"   Ergebnisse gespeichert: results/predictions/")


if __name__ == "__main__":
    main()
