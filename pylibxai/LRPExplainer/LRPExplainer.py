from captum.attr import LRP
from captum.attr import visualization as viz
import numpy as np
from pylibxai.models.GtzanCNN.preprocessing import convert_to_spectrogram
from pylibxai.Interfaces import ViewType
from pylibxai.pylibxai_server import WebView
import matplotlib.pyplot as plt
import torch
import os

class LRPExplainer:
    def __init__(self, model_adapter, context, device, view_type=None):
        predict_fn = model_adapter.get_lrp_predict_fn()
        self.explainer = LRP(predict_fn)
        self.device = device
        self.attribution = None
        self.delta = None
        self.context = context
        self.view_type = view_type
        if view_type == ViewType.WEBVIEW:
            self.view = WebView(context, port=9000)
        elif view_type == ViewType.DEBUG:
            pass

    def explain_instance(self, audio, target, background=None):
        audio = convert_to_spectrogram(audio, self.device)
        audio.requires_grad_(True)
        attributions, delta = self.explainer.attribute(audio, target=target, return_convergence_delta=True)
        return attributions, delta
    
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
    
    def explain(self, audio, target):
        fig, _ = self.explain_instance_visualize(audio, target=target, type="original_image")
        self.context.write_plt_image(fig, os.path.join("lrp", "lrp_spectogram.png"))

        fig, _ = self.explain_instance_visualize(audio, target=target, type="heat_map")
        self.context.write_plt_image(fig, os.path.join("lrp", "lrp_attribution_heat_map.png"))

        attribution = self.get_smoothed_attribution()
        self.context.write_attribution(attribution, os.path.join("lrp", "lrp_attributions.json"))
       
        if self.view_type == ViewType.WEBVIEW:
            self.view.start()
            print('Press Ctrl+C to stop the server.')
            try:
                while True:
                    pass  # Keep the server running
            except KeyboardInterrupt:
                print("Shutting down the server...")
                self.view.stop()
                print("Server stopped.")
