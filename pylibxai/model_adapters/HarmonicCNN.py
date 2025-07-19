import sys
import os
from argparse import Namespace

import torch
from torch.autograd import Variable
import numpy as np
from typing import Callable

from pathlib import Path
from pylibxai.Interfaces import LrpAdapter, LimeAdapter, ShapAdapter, ModelLabelProvider

path_sota = str(Path.home() / 'Desktop' / 'pylibxai' / 'pylibxai' / 'models' / 'sota-music-tagging-models')
sys.path.append(path_sota)
sys.path.append(os.path.join(path_sota, 'training'))
from training.eval import Predict  # can only be imported after appending path_sota in sota_utils

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']

class HarmonicCNN(LimeAdapter, ShapAdapter, ModelLabelProvider):
    def __init__(self, device='cuda'):
        """Harmonic CNN model adapter for music tagging.
        """
        assert device in ['cpu', 'cuda']
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        path_models = os.path.join(path_sota, 'models')

        self.label_to_id = {i: v for i, v in enumerate(TAGS)}
        self.id_to_label = {v: i for i, v in enumerate(TAGS)}

        config = Namespace()
        config.dataset = "jamendo"  # we use the model trained on MSD
        config.model_type = "hcnn"
        config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
        config.input_length = 5 * 16000
        config.batch_size = 1  # we analyze one chunk of the audio
        self.model = Predict.get_model(config)
        
        self.model_state = torch.load(config.model_load_path, map_location=self.device)
        self.model.cuda()
        self.config = config
    
    def get_label_mapping(self):
        """Returns the label mapping for the model."""
        return self.label_to_id

    def map_target_to_id(self, target: str) -> int:
        if target in self.label_to_id:
            return self.label_to_id[target]
        else:
            raise ValueError(f"Target '{target}' not found in label mapping.")

    def get_shap_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        self.model.load_state_dict(self.model_state)
        self.model.cuda()
        self.model.eval()

        def predict_fn(x):
            # Make sure input requires gradients for SHAP
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
            
            # Move to GPU if not already there
            if x.device.type != 'cuda':
                x = x.cuda()
                
            # Forward pass through the model
            output_dict = self.model(x)
            
            output_tensor = output_dict
            print(f'SHAP output_tensor shape: {output_tensor.shape}, type: {type(output_tensor)}')
            return output_tensor

        return predict_fn
    
    def shap_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def get_lrp_predict_fn(self) -> torch.nn.Module:
        class HarmonicCNNWrapper(torch.nn.Module):
            def __init__(self, model, device, model_state):
                super(HarmonicCNNWrapper, self).__init__()
                self.model_state = model_state
                self.model = model 
                self.device = device

            def forward(self, x):
                self.model.load_state_dict(self.model_state)
                self.model.cuda()
                self.model.eval()

                # Make sure input requires gradients for SHAP
                if not x.requires_grad:
                    x = x.detach().clone().requires_grad_(True)

                if x.device.type != 'cuda':
                    x = x.cuda()

                output_dict = self.model(x)
                output_tensor = output_dict
                return output_tensor

        return HarmonicCNNWrapper(self.model, self.device, self.model_state)

    def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        self.model.load_state_dict(self.model_state)
        self.model.cuda()
        self.model.eval()

        def predict_fn(x_array):
            # based on code from sota repo
            audio = torch.zeros(len(x_array), self.config.input_length)
            for i in range(len(x_array)):
                audio[i] = torch.Tensor(x_array[i]).unsqueeze(0)
            # audio as an input tensor is created from all
            # audio slices passed as the function's argument.
            audio = audio.cuda()
            audio = Variable(audio)
            output_dict = self.model(audio) # inference here (input is passed into model)
            output_tensor = output_dict.detach().cpu().numpy()
            return np.array(output_tensor)

        return predict_fn
