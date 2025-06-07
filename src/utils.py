# src/utils.py
"""
Hilfsfunktionen für Visualisierungen.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from . import config

def plot_feature_distribution(feature_df: pd.DataFrame):
    """
    Erstellt und speichert Plots für alle Merkmalsverteilungen.
    """
    print("Erstelle Plots zur Merkmalsverteilung...")
    
    feature_df['status'] = feature_df['label'].map({0: 'Intakt', 1: 'Defekt'})
    
    # Automatische Erkennung aller Merkmalspalten (außer 'label' und 'status')
    feature_cols = [col for col in feature_df.columns if col not in ['label', 'status']]
    
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features))
    # Falls es nur ein Merkmal gibt, wird axes nicht als Array zurückgegeben
    if n_features == 1:
        axes = [axes]
        
    fig.suptitle('Vergleich der Merkmalsverteilungen', fontsize=18, y=1.0)

    for i, col in enumerate(feature_cols):
        sns.histplot(data=feature_df, x=col, hue='status', kde=True, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Verteilung für: {col}', fontsize=12)
        axes[i].set_xlabel('Wert')
        axes[i].set_ylabel('Anzahl')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(config.FEATURE_PLOT_PATH)
    print(f"Plots wurden unter '{config.FEATURE_PLOT_PATH}' gespeichert.")
    plt.close()