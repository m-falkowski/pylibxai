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
    
    def write_plt_image(self, fig, suffix):
        fig.savefig(os.path.join(self.workdir, suffix), bbox_inches='tight')
    
    def write_attribution(self, smoothed_attribution, suffix):
        path = os.path.join(self.workdir, suffix) #"shap_attributions.json")
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
            