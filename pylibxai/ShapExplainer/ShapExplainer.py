import shap

class ShapExplainer:
    def __init__(self):
        pass
    def explain_instance(self, predict_fn, audio, background=None):
        # User supplied function that takes a matrix of samples (# samples x # features):
        # [
        #    s0: [f0, f1, ..., fN]
        #    s1: [f0, f1, ..., fN]
        #    ...
        #    sN: [f0, f1, ..., fN]
        # ]
        # and computes the output of the model for those samples.
        # The output can be a vector (# samples) or a matrix (# samples x # model outputs).
        audio = audio
        explainer = shap.DeepExplainer(predict_fn, background)
        shap_values = explainer.shap_values(audio)
        return shap_values
    