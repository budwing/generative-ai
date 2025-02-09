import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
array, sampling_rate = librosa.load(librosa.ex("trumpet"))

# 计算梅尔时频谱并转换为分贝
mel = librosa.feature.melspectrogram(y=array,sr=sampling_rate,
n_mels=80,fmax=8000)
print(array.shape, mel.shape)
mel_db = librosa.power_to_db(mel, ref=np.max)
plt.figure().set_figwidth(10)
librosa.display.specshow(mel_db,x_axis="time",y_axis="mel",
sr=sampling_rate,fmax=8000)
plt.colorbar()
plt.show()