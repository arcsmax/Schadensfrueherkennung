# src/deep_learning/model.py
"""
Definiert die Architektur des 1D-Convolutional Neural Network (1D-CNN).
"""
import torch
import torch.nn as nn

# Verbesserte CNN-Architektur
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes, signal_length=20480):
        super(Simple1DCNN, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Bestimme die Größe des Feature-Vektors dynamisch
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, signal_length)
            n_features = self.features(dummy_input).view(1, -1).size(1)

        # Classifier mit Dropout zur Regularisierung
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x
