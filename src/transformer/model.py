# src/transformer/model.py
"""
Definiert die Architektur eines Patch-basierten Time Series Transformers (PatchTST-inspiriert).
"""
import torch
import torch.nn as nn
from .. import config

class PatchEmbedding(nn.Module):
    """Wandelt das Rohsignal in eingebettete Patches um."""
    def __init__(self, patch_length, d_model):
        super().__init__()
        self.patch_length = patch_length
        self.projection = nn.Linear(patch_length, d_model)

    def forward(self, x):
        # x shape: [batch_size, 1, signal_length]
        # Teile das Signal in Patches auf
        x = x.unfold(dimension=-1, size=self.patch_length, step=self.patch_length)
        # x shape: [batch_size, 1, num_patches, patch_length]
        x = x.squeeze(1) # -> [batch_size, num_patches, patch_length]
        
        # Projiziere die Patches in den Einbettungsraum
        x = self.projection(x) # -> [batch_size, num_patches, d_model]
        return x

class TimeSeriesTransformer(nn.Module):
    """Ein einfacher Time Series Transformer für die Klassifikation."""
    def __init__(self, num_classes, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config.PATCH_LENGTH, d_model)
        
        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Klassifikationskopf
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embedding(x)
        
        # 2. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 3. Aggregiere die Patch-Ausgaben (Mittelwertbildung ist ein gängiger Ansatz)
        x = x.mean(dim=1)
        
        # 4. Klassifikation
        x = self.classification_head(x)
        return x
