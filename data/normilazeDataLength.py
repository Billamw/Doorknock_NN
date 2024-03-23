import os
import soundfile as sf
import numpy as np

source_dir = 'source_data'  # Pfad zu Ihren Audiodateien
target_length = 2 * 44100  # 8 Sekunden bei einer Abtastrate von 44100

for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)
    if not filename.endswith(".wav"):                       
        
        continue

    # Audiodatei mit soundfile laden
    data, samplerate = sf.read(filepath)

    # Audio auf eine Länge von 8 Sekunden bringen
    if len(data) < target_length:
        data = np.pad(data, (0, target_length - len(data)))  # Mit Nullen auffüllen
    else:
        data = data[:target_length]  # Abschneiden

    # Audio speichern
    sf.write('path_to_output_directory/' + filename, data, samplerate)