from piper.voice import PiperVoice
import wave
import time
import urllib.request, json


def tts():
    # 加载模型（.onnx 和 .onnx.json 必须在同一目录）
    voice = PiperVoice.load("en_US-lessac-medium.onnx")

    text = "Hello! This is Piper TTS running locally on Windows. No internet required."

    with wave.open("piper.wav", "w") as wav_file:
        # 必须在 synthesize 之前手动设置 WAV 头
        wav_file.setnchannels(1)                         # 单声道
        wav_file.setsampwidth(2)                         # 16-bit = 2 bytes
        wav_file.setframerate(voice.config.sample_rate)  # 从模型配置读取采样率
        voice.synthesize(text, wav_file)

    print("Done: piper.wav")

def download_voice(lang, region, name, quality, save_dir="."):
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    filename = f"{lang}_{region}-{name}-{quality}"
    for ext in [".onnx", ".onnx.json"]:
        url = f"{base}/{lang}/{lang}_{region}/{name}/{quality}/{filename}{ext}"
        out = f"{save_dir}/{filename}{ext}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, out)

def list_voices():
    url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/voices.json"
    with urllib.request.urlopen(url) as r:
        voices = json.load(r)

    # 只看英文音色
    for key, v in voices.items():
        if key.startswith("en_"):
            print(f"{key:45s} 语言={v['language']['name_english']:10s} 质量={v['quality']}")

if __name__ == "__main__":
    # Download the voice files (only need to do this once)
    # download_voice("en", "US", "lessac", "medium")

    start_time = time.time()
    tts()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # list_voices()