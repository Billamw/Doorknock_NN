import torch
import torchaudio
import pyaudio
import numpy as np
import librosa
import time
from AudioUtil import AudioUtil

print(torch.__version__)

model = torch.load("data/models/V8_model.pth")

# Audio-Parameter
CHUNK = 1050  # Blockgröße | n teile von 44100
FORMAT = pyaudio.paInt16  # Datenformat
CHANNELS = 2  # Anzahl der Kanäle
RATE = 44100  # Abtastrate





print("Starte Klopfen-Erkennung...")
while True:
    p = pyaudio.PyAudio()

    print("Öffne Mikrofon...")
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
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
    
    # -- new added code --
    aud = torch.from_numpy(np_data).float()

    reaud = AudioUtil.resample(aud, RATE)
    rechan = AudioUtil.rechannel(reaud, CHANNELS)

    dur_aud = AudioUtil.pad_trunc(rechan, 2000)
    sgram = AudioUtil.spectro_gram(sgram, n_mels=64, n_fft=1024, hop_len=None)
    # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    prediction = model(sgram)

    # Wenn Klopfen erkannt wird, Signal an Kopfhörer senden
    if prediction > 0.5:
        # Senden Sie ein Signal an Ihre Kopfhörer
        # ...
        print("Klopfen erkannt!")