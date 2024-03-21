import librosa
import numpy as np
import os

# Definieren der Variablen
source_dir = "path_to_output_directory"

# Durchlaufen des Ordners und Vorverarbeiten der Audiodateien
data = []
labels = []
for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)
    
    # Ignorieren von Verzeichnissen
    if os.path.isdir(filepath):
        continue
    
    # Audio laden
    y, sr = librosa.load(filepath)

    # Berechnen von MFCCs und Mel-Spektrogramm
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    # Daten und Label speichern
    data.append([mfcc, mel])
    labels.append(1 if filename.startswith("knock") else 0)

import pandas as pd

# Pandas DataFrame erstellen
df = pd.DataFrame(data, columns=["mfcc", "mel"])
df["label"] = labels

df.to_csv("audio_data.csv", index=False)