# src/deep_learning/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .. import config

class VibrationDataset(Dataset):
    """Lädt Schwingungsdaten und deren Labels aus der Metadaten-Datei."""
    def __init__(self, metadata_df: pd.DataFrame, is_training: bool = True):
        self.metadata = metadata_df
        # KORREKTUR 1: signal_length wird nun korrekt aus der Konfiguration geholt und als Attribut gespeichert
        self.signal_length = config.SIGNAL_LENGTH 
        self.is_training = is_training
        
        # Erstelle ein Mapping von Label zu Index (z.B. Level 0 -> 0, Level 1 -> 1, ...)
        # Dies wird nun außerhalb gemacht, um konsistent zu sein
        self.label_map = {label: i for i, label in enumerate(sorted(self.metadata['pitting_level'].unique()))}
        self.class_names = [str(label) for label in sorted(self.label_map.keys())]


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Stelle sicher, dass der Index auf die korrekte Zeile im DataFrame verweist
        row = self.metadata.iloc[idx]
        filepath = row['filepath']
        
        data = pd.read_csv(filepath, sep=r'\s+', header=None, names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
        
        # Die Zeile, die vorher den Fehler verursacht hat, funktioniert nun, da self.signal_length existiert
        signal = data['acc_x'].values[:self.signal_length].astype(np.float32)

        # Normalisierung pro Instanz
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = (signal - mean) / std

        signal_tensor = torch.from_numpy(signal).unsqueeze(0)
        
        original_label = row['pitting_level']
        label = self.label_map[original_label]
        
        return signal_tensor, label

    def get_num_classes(self):
        """Gibt die Anzahl der einzigartigen Klassen im Datensatz zurück."""
        return len(self.metadata['pitting_level'].unique())


def create_dataloaders(metadata_path, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SIZE):
    """Erstellt die DataLoader für Training und Validierung."""
    try:
        full_metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        logging.error(f"Metadaten-Datei nicht gefunden unter {metadata_path}. Bitte Pipeline neu starten.")
        return None, None, None

    # Teile die Indizes in Training und Validierung auf (stratifiziert)
    train_indices, val_indices = train_test_split(
        range(len(full_metadata)),
        test_size=validation_split,
        stratify=full_metadata['pitting_level'],
        random_state=config.RANDOM_STATE
    )

    train_metadata = full_metadata.iloc[train_indices]
    val_metadata = full_metadata.iloc[val_indices]

    # Erstelle die Datasets
    train_dataset = VibrationDataset(metadata_df=train_metadata, is_training=True)
    val_dataset = VibrationDataset(metadata_df=val_metadata, is_training=False)
    
    # Erstelle die DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # KORREKTUR 2: Rufe die Methode get_num_classes() mit Klammern auf
    num_classes = train_dataset.get_num_classes()

    return train_loader, val_loader, num_classes