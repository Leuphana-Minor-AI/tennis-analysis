#!/usr/bin/env python3
"""
Konvertiert Roboflow NDJSON Dataset zu YOLO-Pose Format
"""

import os
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve
import urllib.error
import time

def parse_ndjson_file(ndjson_path):
    """
    Parst NDJSON Datei und gibt Metadaten und Bilder zurück
    """
    metadata = None
    images = []
    
    try:
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if data.get('type') == 'dataset':
                        metadata = data
                    elif data.get('type') == 'image':
                        images.append(data)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Fehler beim Parsen von Zeile {line_num + 1}: {e}")
                    continue
    except FileNotFoundError:
        print(f"[ERROR] Datei nicht gefunden: {ndjson_path}")
        return None, None
    
    return metadata, images


def download_image(url, save_path, max_retries=3):
    """
    Lädt ein Bild herunter
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            urlretrieve(url, save_path)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"  [WARN] Download fehlgeschlagen (Versuch {retry_count}/{max_retries}), versuche erneut...")
                time.sleep(2)
            else:
                print(f"  [ERROR] Download fehlgeschlagen nach {max_retries} Versuchen: {e}")
                return False
    return False


def convert_pose_to_yolo(annotations, img_width, img_height, kpt_shape):
    """
    Konvertiert Pose-Annotationen zu YOLO-Format
    YOLO Pose Format: class_id x_center y_center width height kpt1_x kpt1_y kpt1_conf ... kptN_x kptN_y kptN_conf
    
    Nimmt nur die konfigurierten Keypoints gemäß kpt_shape[0]
    """
    if not annotations or 'pose' not in annotations:
        return None
    
    poses = annotations['pose']
    if not poses:
        return None
    
    # Konfigurierte Keypoint-Anzahl
    num_keypoints = kpt_shape[0] if kpt_shape else 14
    
    lines = []
    for pose_data in poses:
        # pose_data ist: [class_id, x1, y1, conf1, x2, y2, conf2, ...]
        if len(pose_data) < 1:
            continue
        
        class_id = int(pose_data[0])
        keypoints = pose_data[1:]  # Rest sind die Keypoints
        
        # Extrahiere genau num_keypoints mit je 3 Werten (x, y, conf)
        kpts_xy = []
        kpts_full = []
        
        for i in range(num_keypoints):
            idx = i * 3
            if idx + 2 < len(keypoints):
                x_norm = float(keypoints[idx])
                y_norm = float(keypoints[idx+1])
                conf = float(keypoints[idx+2])
                
                kpts_xy.append((x_norm, y_norm))
                kpts_full.append((x_norm, y_norm, conf))
        
        # Skip wenn nicht genügend Keypoints
        if len(kpts_full) < num_keypoints:
            continue
        
        # Berechne BBox aus Keypoints
        x_coords = [pt[0] for pt in kpts_xy]
        y_coords = [pt[1] for pt in kpts_xy]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # BBox center und größe (bereits normalisiert)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Klammerung auf [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        # Schreibe YOLO Pose Format
        line_parts = [str(class_id), str(x_center), str(y_center), str(width), str(height)]
        
        # Füge alle num_keypoints hinzu (x, y, conf für jeden Keypoint)
        for x, y, conf in kpts_full[:num_keypoints]:
            line_parts.extend([str(max(0, min(1, x))), 
                              str(max(0, min(1, y))), 
                              str(conf)])
        
        lines.append(' '.join(line_parts))
    
    return '\n'.join(lines) if lines else None


def convert_ndjson_to_yolo(ndjson_path, output_dir='data/tennis_keypoints', download_images=True):
    """
    Hauptfunktion für Konvertierung
    """
    print(f"[INFO] Parsingse NDJSON Datei: {ndjson_path}")
    metadata, images = parse_ndjson_file(ndjson_path)
    
    if not metadata or not images:
        print("[ERROR] Konnte NDJSON Datei nicht parsen")
        return False
    
    print(f"[OK] Gefunden: {len(images)} Bilder")
    print(f"   Keypoint-Format: {metadata.get('kpt_shape')}")
    
    kpt_shape = metadata.get('kpt_shape', [14, 3])
    num_keypoints = kpt_shape[0]
    
    # Erstelle Ausgabeverzeichnis
    base_dir = Path(output_dir)
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    for split in ['train', 'val', 'test']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Erstelle Verzeichnisstruktur in {output_dir}")
    
    # Verarbeite Bilder
    downloaded_count = 0
    labeled_count = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for idx, image_data in enumerate(images):
        filename = image_data.get('file')
        url = image_data.get('url')
        split = image_data.get('split', 'train')
        annotations = image_data.get('annotations')
        width = image_data.get('width', 640)
        height = image_data.get('height', 640)
        
        if not filename or split not in split_counts:
            continue
        
        # Downloade Bild wenn gewünscht
        if download_images and url:
            img_path = images_dir / split / filename
            if not img_path.exists():
                print(f"[{idx+1}/{len(images)}] Downloading {filename}...", end='', flush=True)
                if download_image(url, str(img_path)):
                    print(" [OK]")
                    downloaded_count += 1
                else:
                    print(" [FAILED]")
                    continue
            else:
                print(f"[{idx+1}/{len(images)}] {filename} existiert bereits")
                downloaded_count += 1
        
        # Erstelle Label Datei
        if annotations:
            yolo_label = convert_pose_to_yolo(annotations, width, height, kpt_shape)
            if yolo_label:
                label_filename = filename.rsplit('.', 1)[0] + '.txt'
                label_path = labels_dir / split / label_filename
                with open(label_path, 'w') as f:
                    f.write(yolo_label)
                labeled_count += 1
                split_counts[split] += 1
    
    print(f"\n[OK] Konvertierung abgeschlossen!")
    print(f"   Bilder heruntergeladen: {downloaded_count}/{len(images)}")
    print(f"   Labels erstellt: {labeled_count}")
    print(f"   Split-Verteilung: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    
    # Erstelle data.yaml
    data_yaml = {
        'path': str(base_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Eine Klasse: Tennis Court
        'names': {0: 'tennis_court'},
        'kpt_shape': kpt_shape,  # [num_keypoints, 3]
    }
    
    yaml_path = base_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n[INFO] data.yaml erstellt: {yaml_path}")
    print(f"\n[OK] Dataset bereit zum Training!")
    print(f"   Verwende: python scripts/train.py --data {yaml_path} --model yolo26s-pose.pt")
    
    return True


if __name__ == "__main__":
    ndjson_file = 'data/annotated_datasets/tennis-court-keypoint-detection.ndjson'
    output_dir = 'data/tennis_keypoints'
    
    # Optional: Bilder nicht herunterladen wenn sie bereits vorhanden sind
    download = '--no-download' not in sys.argv
    
    if not os.path.exists(ndjson_file):
        print(f"❌ NDJSON Datei nicht gefunden: {ndjson_file}")
        sys.exit(1)
    
    success = convert_ndjson_to_yolo(ndjson_file, output_dir, download_images=download)
    
    sys.exit(0 if success else 1)
