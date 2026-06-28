"""
Inference script for YOLO Pose model on random frames
"""

import sys
from pathlib import Path
import random
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import cv2
import numpy as np


def run_inference(model_path, frames_dir, num_frames=200, output_dir=None):
    """
    Run inference on random frames using trained YOLO pose model
    
    Args:
        model_path: Path to best.pt weights
        frames_dir: Directory containing frames
        num_frames: Number of random frames to process
        output_dir: Directory to save results
    """
    
    # Setup paths
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        print(f"[ERROR] Frames directory not found: {frames_path}")
        return False
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    # Get all frame files
    all_frames = sorted([f for f in frames_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']])
    
    if len(all_frames) < num_frames:
        print(f"[WARNING] Only {len(all_frames)} frames available, requested {num_frames}")
        num_frames = len(all_frames)
    
    # Select random frames
    selected_frames = random.sample(all_frames, num_frames)
    print(f"[OK] Selected {len(selected_frames)} random frames from {len(all_frames)} total")
    
    # Setup output
    if output_dir is None:
        output_dir = Path("results/inference") / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"[OK] Model loaded successfully")
    
    # Run inference
    print(f"\n[INFO] Running inference on {len(selected_frames)} frames...")
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "num_frames": len(selected_frames),
        "results": []
    }
    
    for idx, frame_path in enumerate(selected_frames, 1):
        try:
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"[{idx}/{len(selected_frames)}] ⚠️  Failed to read: {frame_path.name}")
                continue
            
            # Run inference
            result = model(frame, conf=0.25, verbose=False)
            
            # Extract detections
            keypoints = None
            boxes = None
            if len(result) > 0 and result[0].keypoints is not None:
                keypoints = result[0].keypoints.xy.cpu().numpy() if hasattr(result[0].keypoints, 'xy') else None
                if result[0].boxes is not None:
                    boxes = result[0].boxes.xyxy.cpu().numpy()
            
            detection_count = len(result[0].boxes) if result[0].boxes is not None else 0
            keypoint_count = len(keypoints) if keypoints is not None else 0
            
            # Store results
            results_data["results"].append({
                "frame": frame_path.name,
                "detections": detection_count,
                "keypoints": keypoint_count
            })
            
            # Visualize and save
            frame_with_detections = result[0].plot()
            output_frame_path = output_dir / f"result_{idx:04d}_{frame_path.stem}.jpg"
            cv2.imwrite(str(output_frame_path), frame_with_detections)
            
            status = "✓"
            if idx % 20 == 0 or idx == len(selected_frames):
                print(f"[{idx}/{len(selected_frames)}] {status} {frame_path.name}: {detection_count} detections, {keypoint_count} keypoint objects")
        
        except Exception as e:
            print(f"[{idx}/{len(selected_frames)}] ❌ Error processing {frame_path.name}: {str(e)}")
            continue
    
    # Save results summary
    results_file = output_dir / "inference_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    total_detections = sum(r["detections"] for r in results_data["results"])
    total_keypoints = sum(r["keypoints"] for r in results_data["results"])
    
    print(f"\n{'='*60}")
    print(f"[OK] Inference Complete!")
    print(f"{'='*60}")
    print(f"Frames processed: {len(results_data['results'])}/{len(selected_frames)}")
    print(f"Total detections: {total_detections}")
    print(f"Total keypoint detections: {total_keypoints}")
    print(f"Average detections per frame: {total_detections/len(results_data['results']):.2f}")
    print(f"Output directory: {output_dir}")
    print(f"Results file: {results_file}")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run YOLO Pose inference on random frames")
    parser.add_argument("--model", type=str, default="runs/pose/models/tennis_pose/weights/best.pt",
                       help="Path to trained model weights")
    parser.add_argument("--frames", type=str, default="data/frames",
                       help="Directory containing frames")
    parser.add_argument("--num-frames", type=int, default=200,
                       help="Number of random frames to process")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting YOLO Pose Inference")
    print(f"   Model: {args.model}")
    print(f"   Frames dir: {args.frames}")
    print(f"   Num frames: {args.num_frames}")
    print(f"   Output: {args.output or 'results/inference/<timestamp>'}")
    print()
    
    success = run_inference(
        model_path=args.model,
        frames_dir=args.frames,
        num_frames=args.num_frames,
        output_dir=args.output
    )
    
    sys.exit(0 if success else 1)
