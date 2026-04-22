import time
from TTS.api import TTS
import os
os.environ["COQUI_TOS_AGREED"] = "1"

start_time = time.time()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello, how are you doing today?",
    file_path="coqui.wav",
    speaker="Ana Florence",
    language="en"
)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
