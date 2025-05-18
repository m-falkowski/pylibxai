import torchvision
import torchaudio
import torch.nn.functional as F

TRANSFORM = torchvision.transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                         n_fft=1024,
                                         f_min=0.0,
                                         f_max=11025.0,
                                         n_mels=128),
    torchaudio.transforms.AmplitudeToDB()
])

def pad_or_truncate_waveform(wav, target_len):
    current_len = wav.shape[-1]
    if current_len < target_len:
        pad_amt = target_len - current_len
        wav = F.pad(wav, (0, pad_amt))  # pad end with zeros
    elif current_len > target_len:
        wav = wav[:, :target_len]  # truncate
    return wav

def convert_to_spectrogram(input_tensor, device):
    input_tensor = pad_or_truncate_waveform(input_tensor, target_len=22050*30)
    t = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                             n_fft=1024,
                                             f_min=0.0,
                                             f_max=11025.0,
                                             n_mels=128).to(device),
        torchaudio.transforms.AmplitudeToDB().to(device)
    ])
    input_tensor = t(input_tensor).to(device)


    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    return input_tensor
