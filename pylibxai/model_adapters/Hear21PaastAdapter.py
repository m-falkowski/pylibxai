import torch
from hear21passt.base import get_basic_model,get_model_passt
from torch.autograd import Variable

class Hear21PasstAdapter(object):
    def __init__(self, checkpoint_path=None, device='cuda'):
        """Audio tagging inference wrapper.
        """
        
        assert device in ['cpu', 'cuda']
        self.device = device
        # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
        self.model = get_basic_model(mode="logits")
        self.model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476",  n_classes=50)

    def get_predict_fn(self):
        def predict_fn(x_array):
            # based on code from sota repo
            x = torch.zeros(len(x_array), 29 * 16000)
            #x = torch.zeros(len(x_array), 3254510)
            for i in range(len(x_array)):
                x[i] = torch.Tensor(x_array[i]).unsqueeze(0)

            #x = move_data_to_device(x, self.device)

            with torch.no_grad():
                self.model.eval()
                model = self.model.cuda()
                y = model(x, None)
                print(f'output dictionary: {y}')
                return np.array(y)

            return None
        return predict_fn