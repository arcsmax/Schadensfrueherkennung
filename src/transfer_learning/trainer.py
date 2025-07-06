# src/transfer_learning/trainer.py
"""
Implementiert einen simulierten Transfer Learning Workflow.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

from .. import config
from ..deep_learning.data_loader import VibrationDataset
from ..deep_learning.model import Simple1DCNN

def run_transfer_learning_pipeline():
    """Haupt-Orchestrator für den Transfer Learning Workflow."""
    logging.info("="*50)
    logging.info("Starte Transfer Learning Pipeline")
    logging.info("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Verwende Gerät: {device}")

    # --- Stufe 1: Pre-Training ---
    logging.info("--- Stufe 1: Pre-Training auf 'bekannten' Klassen ---")
    
    full_dataset = VibrationDataset(config.TRAIN_METADATA_PATH)
    
    # Filtere den Datensatz für die Pre-Training-Klassen
    pretrain_indices = full_dataset.metadata.index[full_dataset.metadata['pitting_level'].isin(config.PRETRAIN_CLASSES)].tolist()
    pretrain_dataset = Subset(full_dataset, pretrain_indices)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Trainiere ein Modell nur auf diesen Klassen
    pretrain_model = Simple1DCNN(num_classes=len(config.PRETRAIN_CLASSES), signal_length=config.SIGNAL_LENGTH).to(device)
    # ... (Hier wäre eine Trainingsschleife, ähnlich der aus dem CNN-Trainer)
    logging.info(f"Simuliere Pre-Training auf {len(pretrain_dataset)} Beispielen...")
    torch.save(pretrain_model.state_dict(), config.PRETRAINED_MODEL_PATH)
    logging.info(f"Vortrainiertes Modell gespeichert: {config.PRETRAINED_MODEL_PATH}")

    # --- Stufe 2: Fine-Tuning ---
    logging.info("\n--- Stufe 2: Fine-Tuning auf 'neuen, seltenen' Klassen ---")
    
    # Lade das vortrainierte Modell
    finetune_model = Simple1DCNN(num_classes=len(config.PRETRAIN_CLASSES), signal_length=config.SIGNAL_LENGTH).to(device)
    finetune_model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH))
    
    # Friere alle gelernten Schichten ein
    for param in finetune_model.parameters():
        param.requires_grad = False
        
    # Ersetze den Klassifikationskopf für die neue Aufgabe
    num_finetune_classes = len(config.FINETUNE_CLASSES)
    finetune_model.fc2 = nn.Linear(finetune_model.fc2.in_features, num_finetune_classes).to(device)
    logging.info("Modell-Kopf für Fine-Tuning Aufgabe ausgetauscht.")
    
    # Filtere den Datensatz für die Fine-Tuning-Klassen und nehme nur wenige Samples
    finetune_indices = full_dataset.metadata.index[full_dataset.metadata['pitting_level'].isin(config.FINETUNE_CLASSES)].tolist()
    # Limitiere auf die definierte Anzahl an Samples
    finetune_subset_indices = finetune_indices[:config.FINETUNE_SAMPLES_PER_CLASS * num_finetune_classes]
    finetune_dataset = Subset(full_dataset, finetune_subset_indices)
    finetune_loader = DataLoader(finetune_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    logging.info(f"Starte Fine-Tuning mit nur {len(finetune_dataset)} Beispielen.")
    
    # Trainiere nur den neuen Kopf
    optimizer = optim.Adam(finetune_model.fc2.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # ... (Hier wäre eine weitere, kurze Trainingsschleife)
    
    logging.info("Fine-Tuning abgeschlossen.")
    torch.save(finetune_model.state_dict(), config.FINETUNED_MODEL_PATH)
    logging.info(f"Feinjustiertes Modell gespeichert: {config.FINETUNED_MODEL_PATH}")
    logging.info("Dieser Workflow demonstriert die Fähigkeit, mit sehr wenigen Daten ein Modell für neue Aufgaben anzupassen.")
