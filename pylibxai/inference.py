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
from pylibxai.pylibxai_serve.file_serve import run_file_server
from utils import get_install_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GTZAN_MODEL_PATH= get_install_path() / "pylibxai" / "models" / "GtzanCNN" / "gtzan_cnn.ckpt"

def main():
    parser = argparse.ArgumentParser(description="Process a model name and input path.")
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model to use [sota_music, paans, gtzan].")
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

        print('Starting explanation')
        explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                                 predict_fn=adapter.get_predict_fn(),
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

        # print("predicted label:", label)
        # timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        # outdir = root / 'output'
        # outdir.mkdir(parents=True, exist_ok=True)

        # sf.write(str(outdir / f"explanation{timestamp}.wav"), sum(top_components), 16000, 'PCM_24')
        # sf.write(str(outdir / f"original{timestamp}.wav"), spleeter_factorization.data_provider.get_mix(), 16000, 'PCM_24')
    elif args.explainer == "lrp":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)
        
        explainer = LRPExplainer(adapter.lrp_adapter_fn(), DEVICE)
        fig, _ = explainer.explain_instance_visualize(audio, target=label_id)
        fig.savefig(os.path.join(args.workdir, "lrp_attribution.png"), bbox_inches='tight')
        return
    elif args.explainer == "shap":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)

        explainer = ShapExplainer(adapter.shap_adapter_fn(), DEVICE)
        fig, _ = explainer.explain_instance_visualize(audio, target=label_id)
        fig.savefig(os.path.join(args.workdir, "shap_attribution.png"), bbox_inches='tight')
        explainer.save_attributions(os.path.join(args.workdir, "shap_attributions.json"))
        #explainer.save_spectrogram(audio, os.path.join(args.workdir, "spectogram.png"))
        #return
    else:
        print(f'Unknown explanation type: {args.explainer}')
        return

    # copy input audio to workdir
    shutil.copy(args.input, os.path.join(args.workdir, "input.wav"))

    port = 9000
    print(f"Starting file server at http://localhost:{port}/")
    print(f"Files will be served from: {args.workdir}")
    server = run_file_server(directory=args.workdir, port=port)
    print('Press Ctrl+C to stop the server.')
    try:
        while True:
            pass  # Keep the server running
    except KeyboardInterrupt:
        print("Shutting down the server...")
        server.shutdown()
        print("Server stopped.")
    
if __name__ == '__main__':
    main()