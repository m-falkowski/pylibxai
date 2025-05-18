import torch
import torch.nn.functional as F

from pylibxai.models.GtzanCNN.model import CNN
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram

gtzan_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class GtzanPredictor:
    def __init__(self, model_path, device):
        """
        Initializes the Predictor.

        Args:
            model: The PyTorch model instance.
            model_path (str): Path to the model checkpoint file.
            device (torch.device): The device to run inference on.
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        self.label_to_id = {genre: i for i, genre in enumerate(gtzan_genres)}
        self.id_to_label = {i: genre for i, genre in enumerate(gtzan_genres)}

    def load_model(self):
        # Load model
        self.model = CNN(num_classes=len(gtzan_genres))
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def predict(self, input_tensor):
        """
        Predicts the class for a given input tensor.

        Args:
            input_tensor: (torch.Tensor): waveform of shape [1, 1, H, W]

        Returns:
            tuple: (predicted_label (str), confidence (float))
        """
        # Transform
        input_tensor = convert_to_spectrogram(input_tensor, self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            top1_prob, top1_pred = torch.max(probs, dim=1)

        predicted_label = self.id_to_label[top1_pred.item()]
        confidence = top1_prob.item()

        return predicted_label, confidence

