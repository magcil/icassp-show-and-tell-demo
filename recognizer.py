import argparse
import json
import time
import datetime

import warnings
warnings.filterwarnings("ignore")

from utils.utils import query_sequence_search, get_winner, extract_mel_spectrogram, search_index
from models.neural_fingerprinter import Neural_Fingerprinter
from models.beats.beats_model import BEATs_Model

import faiss
import torch
import numpy as np
import pyaudio
import sounddevice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='Json config for online inference.')

    return parser.parse_args()

def recognize_song(y, H, F, s_flag, k):
    # Inference
    tic = time.perf_counter()
    J = int(np.floor((aggregated_buf.size - F) / H)) + 1
    xq = np.stack(
        [extract_mel_spectrogram(aggregated_buf[j * H:j * H + F]).reshape(1, 256, 32) for j in range(J)]
    )
    out = model(torch.from_numpy(xq).to(device))
    inference_time = 1000 * (time.perf_counter() - tic)

    # Retrieval
    tic = time.perf_counter()
    D, I = index.search(out.cpu().numpy(), k)

    if s_flag == 'sequence search':

        idx, score = query_sequence_search(D, I)
        true_idx = search_index(idx=idx, sorted_arr=sorted_arr)
        winner = json_correspondence[str(true_idx)]

        # Determine offset
        offset = (idx - true_idx) * H / F
        m_start, s_start = divmod(offset, 60)
        m_end, s_end = divmod(offset + dur, 60)

        now = datetime.datetime.now().time()
        out_str = f'Time: {now} | Pred: {winner} | Score: {score:.3f} | Offset: {offset} | ' +\
            f'Query time span: {int(m_start)}:{int(s_start):02d}-{int(m_end)}:{int(s_end):02d}'

    elif s_flag == 'majority vote':
        winner, score = get_winner(d=json_correspondence, I=I, D=D, sorted_array=sorted_arr)
        now = datetime.datetime.now().time()
        out_str = f'Time: {now} | SONG NAME: {winner}'

    query_time = 1000 * (time.perf_counter() - tic)
    total = query_time + inference_time
    print(out_str + f' | Inference: {inference_time:.3f} | Query: {query_time:.3f} | Total: {total:.3f}')


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, 'r') as f:
        args = json.load(f)

    # Parse args
    dur = args['duration']
    with open(args['json'], 'r') as f:
        json_correspondence = json.load(f)
        sorted_arr = np.sort(np.array(list(map(int, json_correspondence.keys()))))
    index = faiss.read_index(args['index'])
    index.nprobe = args['nprobes']
    k = args['neighbors']
    
    # Check for music detector
    if args['music_detector']:
        music_detector = BEATs_Model(path_to_checkpoint=args['music_detector_pt'], device='cpu')

    # Search algorithm
    if args['search algorithm'] == 'majority vote':
        s_flag = 'majority vote'
    elif args['search algorithm'] == 'sequence search':
        s_flag = 'sequence search'
    else:
        raise NotImplementedError(
            f'{args["search algorithm"]} not implemented. Choose either majority vote or sequence search'
        )

    device = 'cuda' if torch.cuda.is_available() and args['device'] == 'cuda' else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(args['weights'], map_location="cpu"))

    F, H, FMB = args['SR'], args['Hop size'], args['FMB']

    p = pyaudio.PyAudio()

    stream = p.open(rate=F, channels=1, format=pyaudio.paFloat32, frames_per_buffer=FMB, input=True)

    try:
        model.eval()
        with torch.no_grad():
            while True:
                frames = []
                for i in range(0, int(F / FMB) * dur):
                    data = stream.read(FMB)
                    frames.append(data)
                aggregated_buf = np.frombuffer(b"".join(frames), dtype=np.float32)
                
                # Check if music_detector is present
                if args['music_detector']:
                    _, _, music_label, _ = music_detector.make_inference_with_waveform(aggregated_buf)
                
                if args['music_detector'] and music_label =='Music':
                    recognize_song(aggregated_buf, H, F, s_flag, k)
                elif args['music_detector'] and music_label != 'Music':
                    now = datetime.datetime.now().time()
                    out_str = f'Time: {now} | No Music is Present. | Sound: {music_label}'
                    print(out_str)
                else:
                    recognize_song(aggregated_buf, H, F, s_flag, k)
                
                

    except KeyboardInterrupt:
        print('Stopped!')
        torch.cuda.empty_cache()
        pass
    pass