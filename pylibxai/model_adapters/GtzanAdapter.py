from pylibxai.models.GtzanCNN.eval import GtzanPredictor
from pylibxai.utils import get_install_path
import torch
MODEL_PATH= get_install_path() / "pylibxai" / "models" / "GtzanCNN" / "best_model.ckpt"

class GtzanAdapter(object):
    def __init__(self, model_path, device='cuda'):
        self.predictor = GtzanPredictor(model_path, device)
        self.predictor.load_model()
        self.device = device

    def lrp_adapter_fn(self):
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
