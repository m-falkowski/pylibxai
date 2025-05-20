from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, device):
        self.explainer = IntegratedGradients(model)
        self.device = device

    def explain_instance(self, audio, target, background=None):
        audio = convert_to_spectrogram(audio, self.device)
        audio.requires_grad_(True)
        attributions, delta = self.explainer.attribute(audio, target=target, return_convergence_delta=True)
        return attributions, delta
    
    def explain_instance_visualize(self, audio, target, background=None):
        audio = convert_to_spectrogram(audio, self.device)
        audio.requires_grad_(True)
        attributions, delta = self.explainer.attribute(audio, target=target, return_convergence_delta=True)

        audio = audio.squeeze().detach().cpu().numpy()
        attributions = attributions.squeeze().detach().cpu().numpy()
        attributions = np.expand_dims(attributions, axis=0)  # shape: [1, H, W]
        attributions = np.transpose(attributions, (1, 2, 0))  # shape: [H, W, 1]

        plt.ioff()
        return viz.visualize_image_attr_multiple(attributions,
                                          audio,
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          fig_size=(24,16),
                                          show_colorbar=True,
                                          outlier_perc=50)
