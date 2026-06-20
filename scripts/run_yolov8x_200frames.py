#!/usr/bin/env python3
"""
Run yolov8x inference on 200 randomly selected frames
"""

import os
import sys
import random
import cv2
from pathlib import Path

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

try:
    from ultralytics import YOLO
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


def run_yolov8x_inference(frames_dir="data/frames", num_frames=200, conf_threshold=0.5):
    """
    Run yolov8x inference on randomly selected frames
    
    Args:
        frames_dir: Directory containing extracted frames
        num_frames: Number of random frames to select
        conf_threshold: Confidence threshold
    
    Returns:
        Detection statistics
    """
    
    if not os.path.exists(frames_dir):
        print(f"❌ Fehler: Frames-Verzeichnis nicht gefunden: {frames_dir}")
        return None
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print(f"❌ Fehler: Keine Frames in {frames_dir} gefunden")
        return None
    
    # Select random frames
    if len(frame_files) < num_frames:
        print(f"⚠️  Warnung: Nur {len(frame_files)} Frames vorhanden, aber {num_frames} angefordert")
        selected_frames = frame_files
    else:
        selected_frames = random.sample(frame_files, num_frames)
        selected_frames.sort()
    
    print(f"🎬 Ausgewählte Frames: {len(selected_frames)}")
    print(f"   Beispiele: {selected_frames[0]}, {selected_frames[len(selected_frames)//2]}, {selected_frames[-1]}\n")
    
    model_name = "yolov8x"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Fehler: Modell nicht gefunden: {model_path}")
        return None
    
    print(f"{'='*70}")
    print(f"🤖 Starte Inference mit Modell: {model_name}")
    print(f"{'='*70}\n")
    
    model = YOLO(model_path)
    
    # Create output directory for results
    output_dir = f"results/predictions/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on selected frames
    detection_stats = {
        "total_frames": len(selected_frames),
        "frames_with_detections": 0,
        "total_detections": 0,
        "class_counts": {},
        "detections_per_frame": []
    }
    
    print(f"Verarbeite {len(selected_frames)} Frames...\n")
    
    for i, frame_file in enumerate(selected_frames):
        frame_path = os.path.join(frames_dir, frame_file)
        
        # Use stream=True to prevent RAM accumulation
        results = model.predict(frame_path, conf=conf_threshold, verbose=False, save=False, stream=True)
        result = next(results)
        
        # Save annotated image
        annotated_image = result.plot()
        output_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_path, annotated_image)
        
        num_detections = len(result.boxes)
        
        if num_detections > 0:
            detection_stats["frames_with_detections"] += 1
            detection_stats["total_detections"] += num_detections
            detection_stats["detections_per_frame"].append((frame_file, num_detections))
            
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                detection_stats["class_counts"][class_name] = detection_stats["class_counts"].get(class_name, 0) + 1
        
        if (i + 1) % 20 == 0 or (i + 1) == len(selected_frames):
            progress = (i + 1) / len(selected_frames) * 100
            print(f"   ✓ {i + 1}/{len(selected_frames)} Frames verarbeitet ({progress:.1f}%) - {detection_stats['total_detections']} Detektionen gefunden")
    
    # Print detailed statistics
    print(f"\n✅ Inference mit {model_name} abgeschlossen!\n")
    print(f"📈 Detaillierte Statistik ({model_name}):")
    print(f"   Bearbeitete Frames: {detection_stats['total_frames']}")
    print(f"   Frames mit Detektionen: {detection_stats['frames_with_detections']} ({100*detection_stats['frames_with_detections']/detection_stats['total_frames']:.1f}%)")
    print(f"   Gesamte Detektionen: {detection_stats['total_detections']}")
    
    if detection_stats['total_detections'] > 0:
        avg_detections = detection_stats['total_detections'] / detection_stats['frames_with_detections']
        print(f"   Durchschn. Detektionen pro Frame (mit Detektionen): {avg_detections:.2f}")
    
    print(f"\n   Klassen (nach Häufigkeit):")
    for class_name, count in sorted(detection_stats["class_counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / detection_stats['total_detections'] if detection_stats['total_detections'] > 0 else 0
        print(f"      {class_name}: {count} ({percentage:.1f}%)")
    
    if detection_stats['detections_per_frame']:
        print(f"\n   Top 10 Frames mit meisten Detektionen:")
        top_frames = sorted(detection_stats['detections_per_frame'], key=lambda x: x[1], reverse=True)[:10]
        for frame_name, det_count in top_frames:
            print(f"      {frame_name}: {det_count} Detektionen")
    
    print(f"\n   💾 Annotierte Frames gespeichert: {output_dir}/")
    
    return detection_stats


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
    print("🎾 Tennis-Analyse - yolov8x Inference auf 200 zufälligen Frames")
    print("=" * 70 + "\n")
    
    # Run inference on 200 random frames
    stats = run_yolov8x_inference(
        frames_dir="data/frames",
        num_frames=200,
        conf_threshold=0.5
    )
    
    if not stats:
        print("❌ Fehler bei der Verarbeitung!")
        return
    
    # Cleanup frames after inference
    cleanup_frames("data/frames")
    
    print(f"\n{'='*70}")
    print("✅ Verarbeitung abgeschlossen!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
