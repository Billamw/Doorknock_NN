import torch
import torchaudio
import pyaudio
import numpy as np
import librosa

model = torch.load("doorknock_model.pt")

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    frames_per_buffer=1024,
)


while True:
    # Audiodaten vom Mikrofon lesen
    data = stream.read(1024)

    # Daten in ein Numpy-Array konvertieren
    np_data = np.frombuffer(data, dtype=np.int16)

    # MFCCs berechnen
    mfccs = librosa.feature.mfcc(np_data, sr=22050)

    # Daten in ein Tensor konvertieren
    tensor = torch.from_numpy(mfccs).float()

    # Vorhersage des Modells
    prediction = model(tensor)

    # Wenn Klopfen erkannt wird, Signal an Kopfhörer senden
    if prediction > 0.5:
        # Senden Sie ein Signal an Ihre Kopfhörer
        # ...
        print("Klopfen erkannt!")