"""
Trainingslogik für das Transformer-Modell.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import logging
from tqdm import tqdm
import json

from .model import TimeSeriesTransformer
from ..deep_learning.data_loader import create_dataloaders
from .. import config

def run_transformer_training_pipeline():
    """Haupt-Orchestrator für den Transformer Workflow mit Optimierungen."""
    logging.info("="*50)
    logging.info("Starte optimierte Time Series Transformer Pipeline")
    logging.info("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Verwende Gerät: {device}")

    train_loader, val_loader, num_classes = create_dataloaders(config.TRAIN_METADATA_PATH)
    if train_loader is None: return

    class_names = [str(i) for i in range(num_classes)]
    logging.info(f"Anzahl der Klassen erkannt: {num_classes}")

    model = TimeSeriesTransformer(
        num_classes=num_classes,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        dropout=config.TRANSFORMER_DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # KORREKTUR: Das 'verbose=True' Argument wurde entfernt.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=2)

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_preds = []
    best_labels = []

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        current_preds = []
        current_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                current_preds.extend(predicted.cpu().numpy())
                current_labels.extend(labels.cpu().numpy())

        val_accuracy = 100 * correct / total
        logging.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_preds = current_preds
            best_labels = current_labels
            torch.save(model.state_dict(), config.SAVED_TRANSFORMER_MODEL_PATH)
            logging.info("Neues bestes Transformer-Modell gespeichert.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logging.info(f"Keine Verbesserung seit {config.EARLY_STOPPING_PATIENCE} Epochen. Training wird vorzeitig beendet.")
            break
            
    logging.info("Lade bestes Modell für finalen Report...")
    model.load_state_dict(torch.load(config.SAVED_TRANSFORMER_MODEL_PATH))
    
    report_text = classification_report(best_labels, best_preds, target_names=class_names, zero_division=0)
    logging.info("--- Finaler Transformer Klassifikations-Report ---")
    print(report_text)
    with open(config.TRANSFORMER_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(report_text)

    report_dict = classification_report(best_labels, best_preds, target_names=class_names, zero_division=0, output_dict=True)
    with open(config.TRANSFORMER_CLASSIFICATION_REPORT_JSON_PATH, 'w') as f:
        json.dump(report_dict, f, indent=4)
    logging.info(f"JSON-Report für Transformer gespeichert: {config.TRANSFORMER_CLASSIFICATION_REPORT_JSON_PATH}")
