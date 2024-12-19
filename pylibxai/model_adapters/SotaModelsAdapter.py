import sys
import os
from argparse import Namespace

import torch
from torch.autograd import Variable
import numpy as np

from pathlib import Path

path_sota = str(Path.home() / 'Desktop' / 'pylibxai' / 'pylibxai' / 'models' / 'sota_music_tagging_models')
sys.path.append(path_sota)
sys.path.append(os.path.join(path_sota, 'training'))
from training.eval import Predict  # can only be imported after appending path_sota in sota_utils

class SotaModelsAdapter(object):
    def __init__(self, model_type="fcn", input_length=29*16000, device='cuda'):
        """Audio tagging inference wrapper.
        """
        assert device in ['cpu', 'cuda']
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
        path_models = os.path.join(path_sota, 'models')

        config = Namespace()
        config.dataset = "msd"  # we use the model trained on MSD
        config.model_type = model_type
        config.model_load_path = os.path.join(path_models, config.dataset, config.model_type, 'best_model.pth')
        config.input_length = input_length
        config.batch_size = 1  # we analyze one chunk of the audio
        self.model = Predict.get_model(config)
        
        self.model_state = torch.load(config.model_load_path, map_location=self.device)
        self.config = config

    def get_predict_fn(self):
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
