import os
# 根据你的实际安装路径修改，确保精确指向 .dll 文件
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
# 设置 Hugging Face 镜像源
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
import soundfile as sf
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import io

def transcribe_audio_from_url(audio_url):
    # 1. 加载处理器和模型
    model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    print("正在加载模型和处理器，请稍候...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # 2. 从 URL 获取音频数据
    print(f"正在下载音频: {audio_url}")
    response = requests.get(audio_url)
    response.raise_for_status() # 确保请求成功
    
    # 3. 使用 soundfile 读取音频 (避免 torchcodec 依赖)
    audio_bytes = io.BytesIO(response.content)
    waveform_np, sample_rate = sf.read(audio_bytes)
    
    # 转换为 torch tensor，形状调整为 [channel, time]
    # soundfile 返回形状为 [time, channels] 或 [time] (单声道)
    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
    else:
        waveform = torch.from_numpy(waveform_np.T).float()
        
    # 转换为单声道 (如果音频是多声道)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 重采样到 16000 Hz (Wav2Vec2 的严格要求)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    # 4. 准备模型输入
    # processor 需要 1D 的 numpy array 或 tensor
    input_values = processor(
        waveform.squeeze().numpy(), 
        return_tensors="pt", 
        sampling_rate=target_sample_rate
    ).input_values
    
    # 5. 模型推理
    print("正在进行推理...")
    with torch.no_grad():
        logits = model(input_values).logits
        
    # 6. 解码结果
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription


if __name__ == "__main__":
    # 替换为你实际的音频直接下载链接 (支持 wav, mp3, flac 等常见格式)
    # 注意：链接必须是音频文件的直链，而不是网页。
    test_url = "https://ielts-prod.oss-cn-hangzhou.aliyuncs.com/audio/72646a611d5d4433af9b73c2b3a27344.wav"
    
    try:
        result = transcribe_audio_from_url(test_url)
        print("-" * 50)
        print("识别结果 (IPA 音素):")
        print(result)
        print("-" * 50)
    except Exception as e:
        print(f"发生错误: {e}")