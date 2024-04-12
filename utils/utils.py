from collections import Counter
from typing import Dict
import os

import numpy as np
import librosa


def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = 8000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 256
) -> np.ndarray:

    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # convert to dB for log-power mel-spectrograms
    return librosa.power_to_db(S, ref=np.max)

def query_sequence_search(D, I):
    compensations = []
    for i, idx in enumerate(I):
        compensations.append([(x - i) for x in idx])
    candidates = np.unique(compensations)
    scores = []
    D_flat = D.flatten()
    I_flat = I.flatten()
    for c in candidates:
        idxs = np.where((c <= I_flat) & (I_flat <= c + len(D)))[0]
        scores.append(np.sum(D_flat[idxs]))
    return candidates[np.argmax(scores)], round(max(scores), 4)

def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree


def search_index(idx: int, sorted_arr: np.ndarray):
    candidate_indices = np.where(sorted_arr <= idx)[0]
    return sorted_arr[candidate_indices].max()


def majority_vote_search(d: Dict, I: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    preds = [d[str(search_index(idx, sorted_array))] for idx in I_flat]
    c = Counter(preds)
    return c.most_common()[0][0]


def get_winner(d: Dict, I: np.ndarray, D: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    D_flat_inverse = 1 / D.flatten()
    preds = np.array([d[str(search_index(idx, sorted_array))] for idx in I_flat])
    c = Counter(preds)
    winner = c.most_common()[0][0]
    idxs = np.where(preds == winner)[0]
    # num_matches = c.most_common()[0][1]

    D_shape = D.shape[0] * D.shape[1]

    return winner, (1 / D_shape) * D_flat_inverse[idxs].sum()

