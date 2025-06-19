from datetime import datetime
import soundfile as sf
import torch
import argparse
import torchaudio
import shutil
import os

from pylibxai.AudioLoader import RawAudioLoader
from pylibxai.audioLIME import lime_audio, SpleeterFactorization
from pylibxai.LRPExplainer import LRPExplainer
from pylibxai.ShapExplainer.ShapExplainer import ShapExplainer
from pylibxai.model_adapters import SotaModelsAdapter, PannsCnn14Adapter
from pylibxai.model_adapters.GtzanAdapter import GtzanAdapter
from pylibxai.pylibxai_server import WebView
from utils import get_install_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GTZAN_MODEL_PATH= get_install_path() / "pylibxai" / "models" / "GtzanCNN" / "gtzan_cnn.ckpt"

def main():
    parser = argparse.ArgumentParser(description="Process a model name and input path.")
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model to use [sota_music, paans, gtzan].")
    parser.add_argument('-u', '--visualize', action='store_true',
                        help="Enable visualization of audio in browser-based UI.")
    parser.add_argument('-e', '--explainer', type=str, required=True,
                        help="Name of the explainer to use [lime, shap, lrp].")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input file or directory.") 
    parser.add_argument('-w', '--workdir', type=str, required=True,
                        help="Path to the workdir directory.")
    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        print(f'Workdir {args.workdir} does not exist, creating it...')
        os.makedirs(args.workdir)

    if not os.path.exists(os.path.join(args.workdir, "shap")):
        os.makedirs(os.path.join(args.workdir, "shap"))

    if not os.path.exists(os.path.join(args.workdir, "lrp")):
        os.makedirs(os.path.join(args.workdir, "lrp"))

    if not os.path.exists(os.path.join(args.workdir, "lime")):
        os.makedirs(os.path.join(args.workdir, "lime"))

    root = get_install_path()
    datadir = root / "data"
    path_sota = str(root / 'sota-music-tagging-models')
    paans_model = str(root / 'pylibxai' / 'models' / 'audioset_tagging_cnn' / 'Cnn14_mAP=0.431.pth')
    
    if args.model == "sota_music":
        adapter = SotaModelsAdapter(model_type="sample", input_length=29 * 16000, device=DEVICE, dataset='jamendo')
    elif args.model == "paans":
        adapter = PannsCnn14Adapter(checkpoint_path=paans_model, device=DEVICE)
    elif args.model == "gtzan":
        adapter = GtzanAdapter(model_path=GTZAN_MODEL_PATH, device=DEVICE)
    else:
        print('Invalid value for -m/--model argument, available: [sota_music, paans, gtzan].')
        return
    
    print(f'Adapter(): {adapter}')
    
    if args.explainer == "lime":
        audio_loader = RawAudioLoader(args.input)
        spleeter_factorization = SpleeterFactorization(audio_loader,
                                                       n_temporal_segments=10,
                                                       composition_fn=None,
                                                       model_name='spleeter:5stems')

        print('Creating explanation object')
        explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

        print('Starting LIME explanation')
        explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                                 predict_fn=adapter.get_lime_predict_fn(),
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

        print("predicted label:", label)
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        outdir = root / 'output'
        outdir.mkdir(parents=True, exist_ok=True)

        shutil.copy(args.input, os.path.join(args.workdir, "original.wav"))
        sf.write(os.path.join(args.workdir, "lime", f"lime_explanation.wav"), sum(top_components), 16000, 'PCM_24')
        #sf.write(str(outdir / f"original{timestamp}.wav"), spleeter_factorization.data_provider.get_mix(), 16000, 'PCM_24')
        return
    elif args.explainer == "lrp":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)
        
        explainer = LRPExplainer(adapter.get_lrp_predict_fn(), DEVICE)
        fig, _ = explainer.explain_instance_visualize(audio, target=label_id, type="original_image")
        fig.savefig(os.path.join(args.workdir, "lrp", "lrp_attribution.png"), bbox_inches='tight')

        fig, _ = explainer.explain_instance_visualize(audio, target=label_id, type="original_image")
        fig.savefig(os.path.join(args.workdir, "lrp", "lrp_attribution.png"), bbox_inches='tight')
        return
    elif args.explainer == "shap":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)

        explainer = ShapExplainer(adapter.get_shap_predict_fn(), DEVICE)
        fig, _ = explainer.explain_instance_visualize(audio, target=label_id, type="original_image")
        fig.savefig(os.path.join(args.workdir, "shap", "shap_spectogram.png"), bbox_inches='tight')

        fig, _ = explainer.explain_instance_visualize(audio, target=label_id, type="heat_map")
        fig.savefig(os.path.join(args.workdir, "shap", "shap_attribution_heat_map.png"), bbox_inches='tight')

        explainer.save_attributions(os.path.join(args.workdir, "shap_attributions.json"))
        #explainer.save_spectrogram(audio, os.path.join(args.workdir, "spectogram.png"))
        #return
    else:
        print(f'Unknown explanation type: {args.explainer}')
        return

    # copy input audio to workdir
    shutil.copy(args.input, os.path.join(args.workdir, "input.wav"))

    if args.visualize:
        server = WebView(directory=args.workdir, port=9000)
        server.start()
        print('Press Ctrl+C to stop the server.')
        try:
            while True:
                pass  # Keep the server running
        except KeyboardInterrupt:
            print("Shutting down the server...")
            server.stop()
            print("Server stopped.")
    else:
        print("Visualization is disabled.")

if __name__ == '__main__':
    main()