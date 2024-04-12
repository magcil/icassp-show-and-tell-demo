import os
import sys
import json
import argparse
import time

import librosa
import faiss
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from models.neural_fingerprinter import Neural_Fingerprinter
from utils.utils import extract_mel_spectrogram

class FileDataset(Dataset):

    def __init__(self, file, sr, hop_size):
        self.y, self.F = librosa.load(file, sr=sr)
        self.H = hop_size
        self.dur = self.y.size // self.F

        # Extract spectrograms
        self._get_spectrograms()

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return torch.from_numpy(self.spectrograms[idx])

    def _get_spectrograms(self):
        self.spectrograms = []
        J = int(np.floor((self.y.size - self.F) / self.H)) + 1
        for j in range(J):
            S = extract_mel_spectrogram(signal=self.y[j * self.H:j * self.H + self.F])
            self.spectrograms.append(S.reshape(1, *S.shape))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to json config file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, 'r') as f:
        config_file = json.load(f)

    model = Neural_Fingerprinter()
    model.load_state_dict(torch.load(config_file['weights'], map_location='cpu'))
    F = config_file['SR']
    H = config_file['hop_size']
    batch_size = config_file['batch_size']

    with open(config_file['json'], 'r') as f:
        json_correspondence = json.load(f)

    index = faiss.read_index(config_file['index'])
    faiss_indexes = sorted([int(x) for x in json_correspondence.keys()])
    next_index = index.ntotal

    youtube_links = config_file['youtube']
    names = config_file['song_names']

    wavs = []
    model.eval()
    with torch.no_grad():
        for name, url in tqdm(zip(names, youtube_links), total=len(names)):
            temp_path = 'temp'

            command = f'yt-dlp -x --audio-format wav --audio-quality 0 --force-overwrites ' +\
                    f'--output {temp_path}.wav ' +\
                    f'--postprocessor-args "-ar {F} -ac 1" ' + url + " --quiet"
            os.system(command)
            time.sleep(5)

            try:
                file_dset = FileDataset(file=temp_path + '.wav', sr=F, hop_size=H)
            except Exception as e:
                print(f'Failed to download {name}')
                raise
            file_dloader = DataLoader(file_dset, batch_size=batch_size, shuffle=False)
            fingerprints = []
            for X in file_dloader:
                X = model(X)
                fingerprints.append(X.numpy())

            fingerprints = np.vstack(fingerprints)
            index.add(fingerprints)
            json_correspondence[next_index] = name
            next_index += fingerprints.shape[0]
            wavs.append(temp_path)

    faiss.write_index(index, config_file['index'])
    with open(config_file['json'], 'w') as f:
        json.dump(json_correspondence, f)