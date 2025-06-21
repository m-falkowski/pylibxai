import torch
import argparse
import torchaudio
import os

from pylibxai.LRPExplainer import LRPExplainer
from pylibxai.ShapExplainer.ShapExplainer import ShapExplainer
from pylibxai.model_adapters import SotaModelsAdapter, PannsCnn14Adapter
from pylibxai.model_adapters.GtzanAdapter import GtzanAdapter
from pylibxai.pylibxai_context import PylibxaiContext
from pylibxai.Explainers import LimeExplainer
from pylibxai.Interfaces import ViewType
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

    context = PylibxaiContext(args.workdir)

    if args.model == "sota_music":
        adapter = SotaModelsAdapter(model_type="fcn", input_length=29 * 16000, device=DEVICE, dataset='jamendo')
    elif args.model == "paans":
        adapter = PannsCnn14Adapter(device=DEVICE)
    elif args.model == "gtzan":
        adapter = GtzanAdapter(model_path=GTZAN_MODEL_PATH, device=DEVICE)
    else:
        print('Invalid value for -m/--model argument, available: [sota_music, paans, gtzan].')
        return
    
    print(f'Adapter(): {adapter}')
    
    view_type = ViewType.WEBVIEW if args.visualize else ViewType.NONE

    # copy input audio to workdir
    context.write_audio(args.input, os.path.join("input.wav"))
    context.write_label_mapping(adapter.get_label_mapping(), os.path.join("labels.json"))
    
    if args.explainer == "lime":
        explainer = LimeExplainer(adapter, context, view_type=view_type)
        explainer.explain(args.input, target=None)
    elif args.explainer == "lrp":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)
        
        explainer = LRPExplainer(adapter, context, DEVICE, view_type=view_type)
        explainer.explain(audio, target=label_id)
    elif args.explainer == "shap":
        audio, _ = torchaudio.load(args.input, normalize=True)
        # extract genre from filename
        genre = args.input.split("/")[-2]
        label_id = adapter.predictor.label_to_id[genre]
        audio = audio.to(DEVICE)

        explainer = ShapExplainer(adapter, context, DEVICE, view_type=view_type)
        explainer.explain(audio, target=label_id)
    else:
        print(f'Unknown explanation type: {args.explainer}')
        return

if __name__ == '__main__':
    main()