import torch
from torch.autograd import Variable
from .panns_inference import Cnn14, labels
import numpy as np

from pylibxai.Interfaces import LimeAdapter, IGradientsAdapter, ModelLabelProvider, LrpAdapter
from utils import get_install_path

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

class Cnn14Adapter(LimeAdapter, IGradientsAdapter, ModelLabelProvider, LrpAdapter):
    def __init__(self, device='cuda'):
        """Audio tagging inference wrapper.
        """
        
        assert device in ['cpu', 'cuda']
        if device == 'cuda':
            assert torch.cuda.is_available()
        self.device = device
        checkpoint_path = str(get_install_path() / 'pylibxai' / 'models' / 'audioset_tagging_cnn' / 'Cnn14_mAP=0.431.pth')
       
        self.label_to_id = {}
        self.id_to_label = {}
        with open(get_install_path() / 'pylibxai' / 'datasets' / 'AudioSet' / 'class_labels_indices.csv', 'r') as f:
            lines = f.readlines()
            self.classes_num = len(lines) - 1
            for line in lines[1:]:
                if '"' in line:
                    parts = line.strip().split(',"')
                    index_mid = parts[0].split(',')
                    index = index_mid[0]
                    display_name = parts[1].rstrip('"')
                else:
                    index, _, display_name = line.strip().split(',')
                self.label_to_id[display_name] = int(index)
                self.id_to_label[int(index)] = display_name

        self.model = Cnn14(sample_rate=32000, window_size=1024, 
            hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
            classes_num=self.classes_num)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')
    
    def get_label_mapping(self):
        """Returns the label mapping for the model."""
        return self.id_to_label
    
    def map_target_to_id(self, target: str) -> int:
        if target in self.label_to_id:
            return self.label_to_id[target]
        else:
            raise ValueError(f"Target '{target}' not found in label mapping.")

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()

        return clipwise_output, embedding
    
    def igrad_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def get_igrad_predict_fn(self):
        def predict_fn(x):
            # Make sure input requires gradients for Integrated Gradients
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
                
            # Move to device while preserving gradient information
            x = move_data_to_device(x, self.device)
            
            # Ensure it requires gradients after moving to device
            if not x.requires_grad:
                x.requires_grad_(True)

            self.model.eval()  # Keep model in eval mode
            output_dict = self.model(x, None)
            
            # Return the output without detaching to preserve gradient information
            return output_dict['clipwise_output']

        return predict_fn

    def get_lrp_predict_fn(self):
        class GtzanNNWrapper(torch.nn.Module):
            def __init__(self, predictor, device):
                super(GtzanNNWrapper, self).__init__()
                self.predictor = predictor
                self.device = device

            def forward(self, x):
                # Make sure input requires gradients for LRP
                if not x.requires_grad:
                    x = x.detach().clone().requires_grad_(True)
                    
                # Move to device while preserving gradient information
                x = move_data_to_device(x, self.device)
            
                # Ensure it requires gradients after moving to device
                if not x.requires_grad:
                    x.requires_grad_(True)

                self.model.eval()  # Keep model in eval mode
                output_dict = self.model(x, None)
            
                # Return the output without detaching to preserve gradient information
                print(f'output_dict keys: {output_dict.keys()}')
                return output_dict['clipwise_output']

        return GtzanNNWrapper(self.model, self.device)

    def get_lime_predict_fn(self, input_length=None):
        length = 5 * 16000 if not input_length else input_length
        def predict_fn(x_array):
            x = torch.zeros(len(x_array), length)
            for i in range(len(x_array)):
                x[i] = torch.Tensor(x_array[i]).unsqueeze(0)

            x = move_data_to_device(x, self.device)

            with torch.no_grad():
                self.model.eval()
                y = self.model(x, None)
                y = y['clipwise_output'].data.cpu().numpy()
                return np.array(y)

            return None
        return predict_fn
