import os
import torch

from google_drive_downloader import GoogleDriveDownloader as gdd
from sample import Sampler


CUDA = 'cuda'
CPU = 'cpu'
SCRIPTED_LM_PATH = './scripted_lm.pt'
SCRIPTED_LM_ID = os.getenv('MODEL_GDD_ID')


class Loader:
    def __init__(self):
        cuda_available = torch.cuda.is_available()
        self.device = torch.device(CUDA if cuda_available else CPU)
        self.path = SCRIPTED_LM_PATH

    def load(self):
        if not os.path.exists(self.path):
            gdd.download_file_from_google_drive(file_id=SCRIPTED_LM_ID, dest_path=self.path)
        model = torch.jit.load(self.path).eval()
        sampler = Sampler(model, self.device)

        return model, sampler
