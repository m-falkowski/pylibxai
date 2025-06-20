import os
import tempfile
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

class PylibxaiContext:
    def __init__(self, workdir):
        self.workdir = workdir
        
        if not os.path.exists(workdir):
            #print(f'Workdir {workdir} exists, using it...')
            os.makedirs(workdir, exist_ok=True)
        elif workdir is None:
            #print(f'Workdir {workdir} does not exist, creating it...')
            self.workdir = tempfile.mkdtemp()
            shutil.copytree(workdir, self.workdir)
            #print(f'Created temp workdir: {self.workdir}')

        if not os.path.exists(os.path.join(self.workdir, "shap")):
            os.makedirs(os.path.join(self.workdir, "shap"))

        if not os.path.exists(os.path.join(self.workdir, "lrp")):
            os.makedirs(os.path.join(self.workdir, "lrp"))

        if not os.path.exists(os.path.join(self.workdir, "lime")):
            os.makedirs(os.path.join(self.workdir, "lime"))
    
    def write_lrp_attribution(self, fig):
        fig.savefig(os.path.join(self.workdir, "lrp", "lrp_attribution.png"), bbox_inches='tight')
    
    def write_shap_attribution_img(self, fig):
        fig.savefig(os.path.join(self.workdir, "shap", "shap_attribution.png"), bbox_inches='tight')
    
    def write_lime_attribution(self, fig):
        fig.savefig(os.path.join(self.workdir, "lime", "lime_attribution.png"), bbox_inches='tight')
    
    def write_shap_spectogram(self, fig):
        fig.savefig(os.path.join(self.workdir, "shap", "shap_spectogram.png"), bbox_inches='tight')

    def write_shap_heat_map(self, fig):
        fig.savefig(os.path.join(self.workdir, "shap", "shap_attribution_heat_map.png"), bbox_inches='tight')

    def write_shap_attribution(self, smoothed_attribution):
        path = os.path.join(self.workdir, "shap_attributions.json")
        with open(path, 'w') as f:
            json.dump({
                "attributions": smoothed_attribution.tolist(),
            }, f, indent=4)
    
    def write_audio(self, audio, suffix, *args, **kwargs):
        audio_path = os.path.join(self.workdir, suffix)
        if isinstance(audio, str):
            shutil.copy(audio, os.path.join(self.workdir, suffix))
        elif isinstance(audio, np.ndarray):
            sf.write(audio_path, audio, *args, **kwargs)
            