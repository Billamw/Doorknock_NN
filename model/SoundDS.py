import os
from torch.utils.data import Dataset
try:
  import AudioUtil as AudioUtil
except:
  import model.AudioUtil as AudioUtil

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, data_path, num_classes=2):
    self.data_path = str(data_path)
    self.files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    self.duration = 2000
    self.sr = 44100
    self.channel = 2
    self.shift_pct = 0.4
    self.num_classes = num_classes
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.files)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):

    audio_file = self.files[idx]

    # Get the Class ID
    if audio_file.split('_')[0] == 'knock':
      class_id = 1
    else:
      class_id = 0
    if self.num_classes == 3:
      if len(audio_file.split('_')) > 1 and audio_file.split('_')[1] == 'with':
        class_id = 2

    aud = AudioUtil.open(os.path.join(self.data_path, audio_file))
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    # return aug_sgram, class_id
    
    return sgram, class_id
