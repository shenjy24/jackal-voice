import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 必须在设置环境变量之后再 import kokoro
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import time

start_time = time.time()

pipeline = KPipeline(lang_code='a')

text = "Hello! This is Kokoro TTS running on Windows."

audio_chunks = []
for result in pipeline(text, voice='af_sarah', speed=1.0):
    audio_chunks.append(result.audio.numpy())

audio = np.concatenate(audio_chunks)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

sf.write('kokoro.wav', audio, 24000)
print("Done! Saved to kokoro.wav")