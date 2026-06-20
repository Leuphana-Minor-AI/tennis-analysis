#!/usr/bin/env python3
"""
Run inference on randomly selected frames
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


def run_inference_on_random_frames(frames_dir="data/frames", num_frames=100, conf_threshold=0.5):
    """
    Run inference on randomly selected frames
    
    Args:
        frames_dir: Directory containing extracted frames
        num_frames: Number of random frames to select
        conf_threshold: Confidence threshold
    
    Returns:
        Dictionary with results for each model
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
    
    models = ["yolov8x", "yolo26s-pose"]
    all_results = {}
    
    for model_name in models:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        
        if not os.path.exists(model_path):
            print(f"❌ Fehler: Modell nicht gefunden: {model_path}")
            continue
        
        print(f"\n{'='*70}")
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
            
            results = model.predict(frame_path, conf=conf_threshold, verbose=False, save=False)
            result = results[0]
            
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
            
            if (i + 1) % 10 == 0 or (i + 1) == len(selected_frames):
                progress = (i + 1) / len(selected_frames) * 100
                print(f"   ✓ {i + 1}/{len(selected_frames)} Frames verarbeitet ({progress:.1f}%) - {detection_stats['total_detections']} Detektionen gefunden")
        
        all_results[model_name] = detection_stats
        
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
            percentage = 100 * count / detection_stats['total_detections']
            print(f"      {class_name}: {count} ({percentage:.1f}%)")
        
        if detection_stats['detections_per_frame']:
            print(f"\n   Top 5 Frames mit meisten Detektionen:")
            top_frames = sorted(detection_stats['detections_per_frame'], key=lambda x: x[1], reverse=True)[:5]
            for frame_name, det_count in top_frames:
                print(f"      {frame_name}: {det_count} Detektionen")
    
    return all_results


def main():
    """
    Main function
    """
    
    print("=" * 70)
    print("🎾 Tennis-Analyse - Inference auf 100 zufälligen Frames")
    print("=" * 70 + "\n")
    
    # Run inference on 100 random frames
    results = run_inference_on_random_frames(
        frames_dir="data/frames",
        num_frames=100,
        conf_threshold=0.5
    )
    
    if not results:
        print("❌ Fehler bei der Verarbeitung!")
        return
    
    # Final comparison
    print(f"\n{'='*70}")
    print("📊 VERGLEICH BEIDER MODELLE")
    print(f"{'='*70}\n")
    
    for model_name, stats in results.items():
        print(f"{model_name}:")
        print(f"  Frames mit Detektionen: {stats['frames_with_detections']}/{stats['total_frames']}")
        print(f"  Gesamte Detektionen: {stats['total_detections']}")
        if stats['total_detections'] > 0:
            print(f"  Klassen gefunden: {', '.join(stats['class_counts'].keys())}")
        print(f"  💾 Annotierte Frames gespeichert: results/predictions/{model_name}/")
        print()
    
    print(f"{'='*70}")
    print("✅ Verarbeitung abgeschlossen!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
