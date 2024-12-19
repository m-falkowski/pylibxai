from datetime import datetime
import librosa
import os
import soundfile as sf
import torch
import argparse

from audioLIME import lime_audio, RawAudioProvider, SpleeterFactorization
from model_adapters import SotaModelsAdapter, PannsCnn14Adapter

from utils import get_install_path

def main():
    parser = argparse.ArgumentParser(description="Process a model name and input path.")
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model to use.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input file or directory.") 
    args = parser.parse_args()

    root = get_install_path()
    datadir = root / "data"
    path_sota = str(root / 'sota-music-tagging-models')
    paans_model = str(root / 'pylibxai' / 'models' / 'audioset_tagging_cnn' / 'Cnn14_mAP=0.431.pth')
    
    if args.model == "sota_music":
        #adapter = SotaModelsAdapter(model_type="fcn", input_length=29 * 16000, device='cuda')
        adapter = SotaModelsAdapter(model_type="hcnn", input_length=5 * 16000, device='cuda')
    elif args.model == "paans":
        adapter = PannsCnn14Adapter(checkpoint_path=paans_model, device='cuda')
    
    print(f'Adapter(): {adapter}')
    
    data_provider = RawAudioProvider(args.input)
    spleeter_factorization = SpleeterFactorization(data_provider,
                                                   n_temporal_segments=10,
                                                   composition_fn=None,
                                                   model_name='spleeter:5stems')

    print('Creating explanation object')
    explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

    print('Starting explanation')
    explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                             predict_fn=adapter .get_predict_fn(),
                                             top_labels=1,
                                             num_samples=16384,
                                             batch_size=16
                                             )

    label = list(explanation.local_exp.keys())[0]
    top_components, component_indeces = explanation.get_sorted_components(label,
                                                                          positive_components=True,
                                                                          negative_components=False,
                                                                          num_components=3,
                                                                          return_indeces=True)

    print("predicted label:", label)
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    outdir = root / 'output'
    outdir.mkdir(parents=True, exist_ok=True) 
    
    sf.write(str(outdir / f"explanation{timestamp}.wav"), sum(top_components), 16000, 'PCM_24')
    sf.write(str(outdir / f"original{timestamp}.wav"), spleeter_factorization.data_provider.get_mix(), 16000, 'PCM_24')

if __name__ == '__main__':
    main()