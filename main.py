import whisper
import torch

audio_path="videos/TikSave.io_7396243763049286919.mp4"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("加载视频")
audio = whisper.load_audio(audio_path)
print("视频加载完成")
audio = whisper.pad_or_trim(audio)

model = whisper.load_model("large-v2",download_root="./whisper_model/")

mel = whisper.log_mel_spectrogram(audio).to(model.device)

options = whisper.DecodingOptions(beam_size=5)

result = whisper.decode(model, mel, options)
print(result.text)