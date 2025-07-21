import torch
import argparse
import torchaudio
import os

from pylibxai.model_adapters import HarmonicCNN, Cnn14Adapter, GtzanCNNAdapter
from pylibxai.pylibxai_context import PylibxaiContext
from pylibxai.Explainers import LimeExplainer, IGradientsExplainer, LRPExplainer
from pylibxai.Interfaces import ViewType, ModelLabelProvider
from utils import get_install_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GTZAN_MODEL_PATH= get_install_path() / "pylibxai" / "models" / "GtzanCNN" / "gtzan_cnn.ckpt"

def main():
    parser = argparse.ArgumentParser(description="Process a model name and input path.")
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model to use [{sota_music, paans, gtzan},]...")
    parser.add_argument('-u', '--visualize', action='store_true',
                        help="Enable visualization of audio in browser-based UI.")
    parser.add_argument('-e', '--explainer', type=str, required=True,
                        help="Name of the explainer to use [lime, integrated-gradients, lrp].")
    parser.add_argument('-t', '--target', type=str, required=True,
                        help="Name or index of the label to explain.\
                              Mapping is done automatically based on the model if the model provides it.") 
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input file or directory.") 
    parser.add_argument('-w', '--workdir', type=str, required=True,
                        help="Path to the workdir directory.")
    parser.add_argument('-p', '--port', type=int, help="Port to use for the web server.")
    parser.add_argument('-d', '--device', type=str, default=DEVICE,
                        help="Device to use for computation [cpu, cuda]. Default is 'cuda' if available, otherwise 'cpu'.")
    args = parser.parse_args()
   
    try:
        port = int(args.port) if args.port else 9000
    except ValueError:
        raise ValueError(f"Invalid port number: {args.port}.")

    device = args.device if args.device is not None else DEVICE
    assert device in ['cpu', 'cuda'], "Device must be either 'cpu' or 'cuda'."
    
    expls = args.explainer.split(",")
    assert all(ex in ["lime", "integrated-gradients", "lrp"] for ex in expls), \
        "Invalid explainer specified. Available options: [lime, integrated-gradients, lrp]."

    context = PylibxaiContext(args.workdir)

    if args.model == "HCNN":
        adapter = HarmonicCNN(device=device)
    elif args.model == "CNN14":
        adapter = Cnn14Adapter(device=device)
    elif args.model == "GtzanCNN":
        adapter = GtzanCNNAdapter(model_path=GTZAN_MODEL_PATH, device=device)
    else:
        print('Invalid value for -m/--model argument, available: [HCNN, CNN14, GtzanCNN].')
        return
    
    view_type = ViewType.WEBVIEW if args.visualize else ViewType.DEBUG
    expl_count = len(expls)

    # Attempt parsing label as an integer, if it fails then assume it's a string label
    try:
        target = int(args.target)
    except ValueError:
        target = args.target

    # copy input audio to workdir
    context.write_audio(args.input, os.path.join("input.wav"))
    if issubclass(type(adapter), ModelLabelProvider):
        context.write_label_mapping(adapter.get_label_mapping(), os.path.join("labels.json"))
    
    if "lime" in expls:
        view = view_type if expl_count == 1 else ViewType.NONE
        expl_count -= 1
        explainer = LimeExplainer(adapter, context, view_type=view, port=port)
        explainer.explain(args.input, target=None)
    if "lrp" in expls:
        view = view_type if expl_count == 1 else ViewType.NONE
        expl_count -= 1
        audio, _ = torchaudio.load(args.input, normalize=True)
        audio = audio.to(device)
        explainer = LRPExplainer(adapter, context, device, view_type=view, port=port)
    if "integrated-gradients" in expls:
        view = view_type if expl_count == 1 else ViewType.NONE
        expl_count -= 1
        audio, _ = torchaudio.load(args.input, normalize=True)
        audio = audio.to(device)
        explainer = IGradientsExplainer(adapter, context, device, view_type=view, port=port)
        explainer.explain(audio, target=target)

if __name__ == '__main__':
    main()