import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import simpleaudio as sa
# 加载小号音频样例
array, sampling_rate = librosa.load(librosa.ex("trumpet"))
print(len(array), sampling_rate)
# 播放音频
play_obj = sa.play_buffer((array * 32767).astype(np.int16), 1, 2, sampling_rate)
# 绘制波形图
plt.figure().set_figwidth(10)
librosa.display.waveshow(array, sr=sampling_rate)
plt.show()