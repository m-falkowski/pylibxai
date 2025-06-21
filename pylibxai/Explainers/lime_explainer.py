from pylibxai.AudioLoader import RawAudioLoader
from pylibxai.audioLIME import lime_audio, SpleeterFactorization
from pylibxai.Interfaces import ViewType
from pylibxai.pylibxai_server import WebView
import os

class LimeExplainer:
    def __init__(self, adapter, context, view_type):
        self.adapter = adapter
        self.context = context
        self.view_type = view_type
        if view_type == ViewType.WEBVIEW:
            self.view = WebView(context, port=9000)
        elif view_type == ViewType.DEBUG:
            pass
        
    def explain(self, audio, target=None): 
        audio_loader = RawAudioLoader(audio)
        spleeter_factorization = SpleeterFactorization(audio_loader,
                                                       n_temporal_segments=10,
                                                       composition_fn=None,
                                                       model_name='spleeter:5stems')

        print('Creating explanation object')
        explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

        print('Starting LIME explanation')
        explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                                 predict_fn=self.adapter.get_lime_predict_fn(),
                                                 top_labels=1,
                                                 num_samples=16384,
                                                 batch_size=16
                                                 )

        label = list(explanation.local_exp.keys())[0]
        top_components, component_indices = explanation.get_sorted_components(label,
                                                                              positive_components=True,
                                                                              negative_components=False,
                                                                              num_components=3,
                                                                              return_indeces=True)
        print("Top components:", component_indices)
        print("Top components shape:", [c.shape for c in top_components])
        print("predicted label:", label)

        self.context.write_audio(audio, os.path.join("lime", "original.wav"))
        self.context.write_audio(sum(top_components), os.path.join("lime", f"lime_explanation.wav"), 16000, 'PCM_24')

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
