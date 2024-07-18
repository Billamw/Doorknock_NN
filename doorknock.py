import torch
import pyaudio
import wave
import os
import model.AudioUtil as AudioUtil

print(torch.__version__)
model = torch.load('data/models/V8_model_fullV2.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

CHUNK = 1050
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record_audio(duration=2):
    frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        # Calculate the correct number of iterations to cover the duration
        num_frames = int((RATE / CHUNK) * duration)
        for _ in range(num_frames):
            data = stream.read(CHUNK)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return frames

def save_temp_audio(frames, filename="temp_audio.wav"):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def process_audio_file(filename):
    class_names = {0: 'Noise', 1: 'Knock', 2: 'kn_se'} 
    aud = AudioUtil.open(filename)
    os.remove(filename)
    # reaud = AudioUtil.resample(aud, RATE)
    # rechan = AudioUtil.rechannel(reaud, CHANNELS)
    # dur_aud = AudioUtil.pad_trunc(rechan, 2000)
    rechan = AudioUtil.rechannel(aud, 2)
    sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
    # sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
    sgram = sgram.to(device)
    model.eval()
    with torch.no_grad():
        inputs = sgram

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        outputs = model(inputs.unsqueeze(0))

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs, 1)
        # Convert predictions and actual labels to class names
        predicted_classes = [class_names.get(p.item(), p.item()) for p in prediction][0]
        print(outputs)
    # print(prediction.item())
    print(predicted_classes)
    # if prediction > 0.5:
    #     print("Klopfen erkannt!")

print("Starte Klopfen-Erkennung...")
while True:
    frames = record_audio()
    save_temp_audio(frames)
    process_audio_file("temp_audio.wav")