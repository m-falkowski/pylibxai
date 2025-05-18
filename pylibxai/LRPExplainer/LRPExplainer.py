from captum.attr import LRP
from captum.attr import visualization as viz
import numpy as np
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram

class LRPExplainer:
    def __init__(self, model, device):
        self.explainer = LRP(model)
        self.device = device

    def explain_instance(self, audio, target, background=None):
        audio = convert_to_spectrogram(audio, self.device)
        attributions = self.explainer.attribute(audio, target=target)
        return attributions
    
    def explain_instance_visualize(self, audio, target, background=None):
        import matplotlib.pyplot as plt
        audio = convert_to_spectrogram(audio, self.device)
        attributions = self.explainer.attribute(audio, target=target)

        audio = audio.squeeze().detach().cpu().numpy()

        attributions = attributions.squeeze().detach().cpu().numpy()
        attributions = np.expand_dims(attributions, axis=0)  # shape: [1, H, W]
        attributions = np.transpose(attributions, (1, 2, 0))  # shape: [H, W, 1]

        plt.ioff()
        return viz.visualize_image_attr_multiple(attributions,
                                          audio,
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          show_colorbar=True,
                                          outlier_perc=50)
