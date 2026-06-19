# Konfiguration für Tennis-Detektor
import os

# Roboflow Einstellungen
ROBOFLOW_CONFIG = {
    "workspace": None,  # Setze deinen Workspace
    "project": None,    # Setze dein Projekt
    "version": None,    # Setze die Version
    "api_key": os.getenv("ROBOFLOW_API_KEY"),  # Setze Env-Variable oder direkt hier
}

# YOLOv8 Modellgröße
MODEL_SIZE = "x"  # Options: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)

# Training-Parameter
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "patience": 20,
    "device": 0,  # GPU Nummer, -1 für CPU
    "optimizer": "auto",
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "augment": True,
}

# Pfade
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
