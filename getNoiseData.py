import random
import pyaudio
import wave
import os

# Audio-Parameter
CHUNK = 1024  # Blockgröße
FORMAT = pyaudio.paInt16  # Datenformat
CHANNELS = 1  # Anzahl der Kanäle
RATE = 44100  # Abtastrate

# Dateiname und Pfad
i = len(os.listdir("test_data")) + 1
filename = f"noise_{i}.wav"
filepath = os.path.join("test_data", filename)

# Audioaufnahme initialisieren
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Aufnahme und Speicherung der Audiodaten
frames = []
for i in range(0, int(RATE / CHUNK * random.uniform(5, 7))):
    data = stream.read(CHUNK)
    frames.append(data)

# Aufräumen
stream.stop_stream()
stream.close()
p.terminate()

# Überprüfen, ob das Verzeichnis existiert
dir_name = os.path.dirname(filepath)
if not os.path.exists(dir_name):
    # Wenn nicht, erstellen Sie es
    os.makedirs(dir_name)

# Speichern der Audiodatei
wf = wave.open(filepath, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Audiodatei gespeichert: {filepath}")
