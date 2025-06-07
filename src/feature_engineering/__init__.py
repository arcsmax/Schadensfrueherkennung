# src/feature_engineering/__init__.py
"""
Feature Engineering Paket

Dieses Paket bündelt alle Funktionen zur Extraktion von Merkmalen
aus den Zeit-, Frequenz-, Zeit-Frequenz- und Ordnungs-Domänen.
"""

# Importiert die Haupt-Funktionen, damit sie einfach von außerhalb
# aufgerufen werden können (z.B. aus main.py).
from .main_extractor import extract_features_for_all_files, extract_features_for_single_instance

__all__ = [
    'extract_features_for_all_files',
    'extract_features_for_single_instance'
]