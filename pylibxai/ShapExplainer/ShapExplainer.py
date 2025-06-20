from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram
import matplotlib.pyplot as plt
import json
import torch

class ShapExplainer:
    def __init__(self, model, device):
        self.explainer = IntegratedGradients(model)
        self.device = device
        self.attribution = None
        self.delta = None

    def explain_instance(self, audio, target, background=None):
        audio = convert_to_spectrogram(audio, self.device)
        audio.requires_grad_(True)
        attributions, delta = self.explainer.attribute(audio, target=target, return_convergence_delta=True)
        return attributions, delta
    
    #fig, _ = explainer.explain_instance_visualize(audio, target=label_id, type="original_image")
    def explain_instance_visualize(self, audio, target, type=None, background=None, attr_sign='positive'):
        audio = convert_to_spectrogram(audio, self.device)
        audio.requires_grad_(True)
        attributions, delta = self.explainer.attribute(audio, target=target, return_convergence_delta=True)
        self.attribution = attributions
        self.delta = delta

        audio = audio.squeeze().detach().cpu().numpy()
        attributions = attributions.squeeze().detach().cpu().numpy()
        attributions = np.expand_dims(attributions, axis=0)  # shape: [1, H, W]
        attributions = np.transpose(attributions, (1, 2, 0))  # shape: [H, W, 1]

        plt.ioff()
        return viz.visualize_image_attr(attributions,
                                        audio,
                                        type,
                                        attr_sign,
                                        fig_size=(24,16),
                                        show_colorbar=True,
                                        outlier_perc=50)

    def get_attribution(self):
        return self.attribution, self.delta

    def get_smoothed_attribution(self):
        def moving_average(data, window_size=15):
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')

        attribution = self.attribution.squeeze()
        positive_attribution = torch.clamp(attribution, min=0.0)
        summed_attribution = positive_attribution.sum(dim=0).detach().cpu().numpy()  # Shape: [1292]
        smoothed_attribution = moving_average(summed_attribution)
        return smoothed_attribution

