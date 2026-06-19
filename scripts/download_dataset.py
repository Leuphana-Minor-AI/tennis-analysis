#!/usr/bin/env python3
"""
Datensatz von Roboflow herunterladen und vorbereiten
"""

import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Fehler: roboflow nicht installiert. Bitte 'pip install roboflow' ausführen.")
    sys.exit(1)

from config import ROBOFLOW_CONFIG, DATA_DIR


def download_dataset(api_key, workspace, project, version, format="yolov8"):
    """
    Lädt Datensatz von Roboflow herunter
    
    Args:
        api_key: Roboflow API Key
        workspace: Workspace Name
        project: Projekt Name
        version: Datensatz Version
        format: Export Format (default: yolov8)
    
    Returns:
        Pfad zum heruntergeladenen Datensatz
    """
    
    # Erstelle Datensatz-Verzeichnis
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    print(f"📥 Starte Download von Roboflow...")
    print(f"   Workspace: {workspace}")
    print(f"   Projekt: {project}")
    print(f"   Version: {version}")
    
    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.versions(version).download(format, location=DATA_DIR)
        
        print(f"✅ Datensatz erfolgreich heruntergeladen!")
        print(f"   Pfad: {dataset.location}")
        
        return dataset.location
    
    except Exception as e:
        print(f"❌ Fehler beim Download: {e}")
        return None


def verify_dataset(data_dir):
    """
    Verifiziert die Datensatz-Struktur
    """
    required_files = ["data.yaml", "images", "labels"]
    
    for item in required_files:
        path = os.path.join(data_dir, item)
        if not os.path.exists(path):
            print(f"❌ Fehler: {item} nicht gefunden in {data_dir}")
            return False
    
    print(f"✅ Datensatz-Struktur OK")
    
    # Zeige Statistiken
    train_images = len(os.listdir(os.path.join(data_dir, "images", "train")))
    val_images = len(os.listdir(os.path.join(data_dir, "images", "val")))
    
    print(f"   Training-Bilder: {train_images}")
    print(f"   Validierungs-Bilder: {val_images}")
    
    return True


if __name__ == "__main__":
    # Hole Konfiguration
    api_key = ROBOFLOW_CONFIG.get("api_key")
    workspace = ROBOFLOW_CONFIG.get("workspace")
    project = ROBOFLOW_CONFIG.get("project")
    version = ROBOFLOW_CONFIG.get("version")
    
    # Überprüfe ob Konfiguration vorhanden
    if not all([api_key, workspace, project, version]):
        print("❌ Fehler: Bitte API Key, Workspace, Projekt und Version in config.py setzen!")
        print("\nBeispiel:")
        print('  api_key = "your_api_key"')
        print('  workspace = "your_workspace"')
        print('  project = "tennis_detection"')
        print('  version = 1')
        sys.exit(1)
    
    # Download
    dataset_path = download_dataset(api_key, workspace, project, version)
    
    if dataset_path:
        verify_dataset(dataset_path)
