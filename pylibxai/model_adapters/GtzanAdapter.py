from pylibxai.models.GtzanCNN.eval import GtzanPredictor
from pylibxai.utils import get_install_path
import torch
import numpy as np
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram
from pylibxai.Interfaces import LrpAdapter, LimeAdapter, ShapAdapter, ModelLabelProvider
import torch.nn.functional as F
from typing import Dict
MODEL_PATH = get_install_path() / "pylibxai" / "models" / "GtzanCNN" / "best_model.ckpt"

class GtzanAdapter(LrpAdapter, LimeAdapter, ShapAdapter, ModelLabelProvider):
    def __init__(self, model_path, device='cuda'):
        self.predictor = GtzanPredictor(model_path, device)
        self.predictor.load_model()
        self.device = device
        self.target_length = 22050 * 30  # Expected audio length

    def pad_or_truncate_waveform(self, wav, target_len):
        current_len = wav.shape[-1]
        if current_len < target_len:
            pad_amt = target_len - current_len
            wav = F.pad(wav, (0, pad_amt))  # pad end with zeros
        elif current_len > target_len:
            wav = wav[:, :target_len]  # truncate
        return wav
    
    def get_label_mapping(self) -> Dict[int, str]:
        return self.predictor.label_to_id

    def get_lime_predict_fn(self):
        self.predictor.model.eval()

        def predict_fn(x_array):
            # Convert numpy array to tensor and ensure correct shape
            audio = torch.zeros(len(x_array), self.target_length, device=self.device)
            for i in range(len(x_array)):
                audio_tensor = torch.Tensor(x_array[i]).unsqueeze(0).to(self.device)
                audio_tensor = self.pad_or_truncate_waveform(audio_tensor, self.target_length)
                audio[i] = audio_tensor.squeeze(0)

            # Convert to spectrogram with proper shape [batch_size, 1, mel_bins, time]
            audio = convert_to_spectrogram(audio, self.device)

            # Ensure correct shape for batch normalization
            if audio.dim() == 3:
                audio = audio.unsqueeze(1)  # Add channel dimension if missing

            with torch.no_grad():
                output_dict = self.predictor.model(audio)
            output_tensor = output_dict.detach().cpu().numpy()
            return np.array(output_tensor)

        return predict_fn

    def get_lrp_predict_fn(self):
        class GtzanNNWrapper(torch.nn.Module):
            def __init__(self, predictor, device):
                super(GtzanNNWrapper, self).__init__()
                self.predictor = predictor
                self.device = device

            def forward(self, x):
                x.requires_grad_(True)
                self.predictor.model.eval()
                return self.predictor.model(x)

        return GtzanNNWrapper(self.predictor, self.device)

    def get_shap_predict_fn(self):
        def shap_fn(x):
            x.requires_grad_(True)
            self.predictor.model.eval()
            return self.predictor.model(x)
        return shap_fn
