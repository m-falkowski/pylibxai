from datetime import datetime
import librosa
import os
import soundfile as sf
import torch
import argparse

from pylibxai.AudioLoader import RawAudioLoader
from pylibxai.audioLIME import lime_audio, SpleeterFactorization
from pylibxai.ShapExplainer import ShapExplainer
from pylibxai.model_adapters import SotaModelsAdapter, PannsCnn14Adapter

from utils import get_install_path

def main():
    parser = argparse.ArgumentParser(description="Process a model name and input path.")
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model to use [sota_music, paans].")
    parser.add_argument('-e', '--explainer', type=str, required=True,
                        help="Name of the explainer to use [lime, shap].")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input file or directory.") 
    args = parser.parse_args()

    root = get_install_path()
    datadir = root / "data"
    path_sota = str(root / 'sota-music-tagging-models')
    paans_model = str(root / 'pylibxai' / 'models' / 'audioset_tagging_cnn' / 'Cnn14_mAP=0.431.pth')
    
    if args.model == "sota_music":
        adapter = SotaModelsAdapter(model_type="fcn", input_length=29 * 16000, device='cuda', dataset='jamendo')
    elif args.model == "paans":
        adapter = PannsCnn14Adapter(checkpoint_path=paans_model, device='cuda')
    else:
        print('Invalid value for -m/--model argument, available: [sota_music, paans].')
        return
    
    print(f'Adapter(): {adapter}')
    
    audio_loader = RawAudioLoader(args.input)
    if args.explainer == "lime":
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
    elif args.explainer == "shap":
        print('SHAP explanation')
        audio = audio_loader.initialize_mix()
        audio = torch.tensor(audio).cuda().unsqueeze(0)
        background = torch.zeros((1, len(audio))).to(device='cuda')
        print(f'{background.shape}')

        explainer = ShapExplainer()
        shap_values = explainer.explain_instance(adapter.model, audio, background=background)
        print(f"SHAP values for the input audio: {shap_values}")
        return
    else:
        print(f'Unknown explanation type: {args.explainer}')
        return

    print("predicted label:", label)
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    outdir = root / 'output'
    outdir.mkdir(parents=True, exist_ok=True) 
 
    sf.write(str(outdir / f"explanation{timestamp}.wav"), sum(top_components), 16000, 'PCM_24')
    sf.write(str(outdir / f"original{timestamp}.wav"), spleeter_factorization.data_provider.get_mix(), 16000, 'PCM_24')
    
if __name__ == '__main__':
    main()