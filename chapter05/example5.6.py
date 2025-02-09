import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.displa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))
# 计算短时傅里叶变换并转换为分贝
stft = librosa.stft(array)
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
print(array.shape, stft.shape)
plt.figure().set_figwidth(12)
librosa.display.specshow(stft_db, x_axis="time", y_axis="hz")
plt.colorbar()
plt.show()