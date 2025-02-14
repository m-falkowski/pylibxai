import torch
from torch.autograd import Variable
from .panns_inference import Cnn14, labels
import numpy as np

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

class PannsCnn14Adapter(object):
    def __init__(self, checkpoint_path=None, device='cuda'):
        """Audio tagging inference wrapper.
        """
        if not checkpoint_path:
            raise RuntimeError("Checkpoint path required")
        
        assert device in ['cpu', 'cuda']
        if device == 'cuda':
            assert torch.cuda.is_available()
        self.device = device
        
        self.labels = labels
        self.classes_num = len(labels) #classes_num # len(labels)
        print(f'{self.classes_num=}')

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

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()

        return clipwise_output, embedding

    def get_predict_fn(self, input_length=None):
        length = 29 * 16000 if not input_length else input_length
        def predict_fn(x_array):
            # based on code from sota repo
            x = torch.zeros(len(x_array), length)
            #x = torch.zeros(len(x_array), 3254510)
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
