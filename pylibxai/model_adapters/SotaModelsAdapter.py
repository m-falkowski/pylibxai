import sys
import os
from argparse import Namespace

import torch
from torch.autograd import Variable
import numpy as np

from pathlib import Path
from pylibxai.Interfaces import LrpAdapter, LimeAdapter, ShapAdapter, ModelLabelProvider

path_sota = str(Path.home() / 'Desktop' / 'pylibxai' / 'pylibxai' / 'models' / 'sota-music-tagging-models')
sys.path.append(path_sota)
sys.path.append(os.path.join(path_sota, 'training'))
from training.eval import Predict  # can only be imported after appending path_sota in sota_utils

# TODO: LrpAdapter, ShapAdapter should be implemented
class SotaModelsAdapter(LimeAdapter, ShapAdapter, ModelLabelProvider):
    def __init__(self, model_type="fcn", device='cuda', **kwargs):
        """Audio tagging inference wrapper.
        """
        assert device in ['cpu', 'cuda']
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.model_map = {
            'fcn': 29 * 16000,
            'musicnn': 3 * 16000,
            'crnn': 29 * 16000,
            'sample': 59049,
            'se': 59049,
            'attention': 15 * 16000,
            'hcnn': 5 * 16000,
            'short': 59049,
            'short_res': 59049
        }
    
        path_models = os.path.join(path_sota, 'models')

        config = Namespace()
        if kwargs.get('dataset') is not None:
            config.dataset = kwargs['dataset']
        else:
            config.dataset = "msd"  # we use the model trained on MSD
        config.model_type = model_type
        config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
        config.input_length = self.model_map[model_type]  # input length in samples
        config.batch_size = 1  # we analyze one chunk of the audio
        self.model = Predict.get_model(config)
        
        self.model_state = torch.load(config.model_load_path, map_location=self.device)
        self.model.cuda()
        self.config = config
    
    def get_label_mapping(self):
        """Returns the label mapping for the model."""
        #if self.config.dataset == 'msd':
        #    label_mapping = Predict.get_msd_label_mapping()
        #elif self.config.dataset == 'jamendo':
        #    label_mapping = Predict.get_jamendo_label_mapping()
        #else:
        #    raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        return {}
    
    def get_shap_predict_fn(self):
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

    def get_lrp_predict_fn(self):
        class SotaNNWrapper(torch.nn.Module):
            def __init__(self, model, device, model_state):
                super(SotaNNWrapper, self).__init__()
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

        return SotaNNWrapper(self.model, self.device, self.model_state)

    def get_lime_predict_fn(self):
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

def composition_fn(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x
