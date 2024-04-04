import torch
import torchaudio
import pyaudio
import numpy as np
import librosa
import time

model = torch.load("data/models/KnnmodelV2-2.pt")

# Audio-Parameter
CHUNK = 1050  # Blockgröße | n teile von 44100
FORMAT = pyaudio.paInt16  # Datenformat
CHANNELS = 1  # Anzahl der Kanäle
RATE = 44100  # Abtastrate





print("Starte Klopfen-Erkennung...")
while True:
    p = pyaudio.PyAudio()

    print("Öffne Mikrofon...")
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024,
    )
    # Audiodaten vom Mikrofon lesen
    start = time.time()
        # Aufnahme und Speicherung der Audiodaten
    frames = []
    for i in range(0, int(RATE / CHUNK * 2)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Aufräumen
    stream.stop_stream()
    stream.close()
    p.terminate()

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