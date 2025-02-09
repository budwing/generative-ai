from transformers import pipeline
from scipy.io.wavfile import write
import numpy as np

# 创建文本到音频管道，并将文本转换为音频
synthesizer = pipeline(task="text-to-audio")
audio = synthesizer("I love this book. [laughs]")
print(audio)

# 提取音频波形数据和采样率
audio_array = audio["audio"][0]
sampling_rate = audio["sampling_rate"]

# 保存为 .wav 文件
output_file = "output.wav"
write(output_file, sampling_rate, (audio_array * 32767).astype(np.int16))
print(f"Audio saved to {output_file}")