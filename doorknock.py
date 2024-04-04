import torch
import torchaudio
import pyaudio
import numpy as np
import librosa
import time

model = torch.load("data/models/KnnmodelV2-2.pt")

p = pyaudio.PyAudio()

print("Öffne Mikrofon...")
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    frames_per_buffer=1024,
)

print("Starte Klopfen-Erkennung...")
while True:
    # Audiodaten vom Mikrofon lesen
    start = time.time()
    data = stream.read(3*1024)
    end = time.time()
    print(f"Time taken: {end-start}")

    np_data = np.frombuffer(data, dtype=np.float32)

    # Überprüfen, ob die Daten endliche Werte enthalten
    if not np.all(np.isfinite(np_data)):
        print("Warning: Audio data contains non-finite values. Replacing with zeros.")
        np_data = np.nan_to_num(np_data)

    spectrum = librosa.feature.melspectrogram(y=np_data, sr=44100, n_mels=128, fmax=8000)
    # flatten the spectrum
    spectrum = spectrum.flatten()
    print(spectrum.shape)
    end = time.time()
    print(f"Time taken: {end-start}")
    # Daten in ein Tensor konvertieren
    tensor = torch.from_numpy(spectrum).float().to("cuda")
    print(tensor.shape)
    # Vorhersage des Modells
    prediction = model(tensor)

    # Wenn Klopfen erkannt wird, Signal an Kopfhörer senden
    if prediction > 0.5:
        # Senden Sie ein Signal an Ihre Kopfhörer
        # ...
        print("Klopfen erkannt!")