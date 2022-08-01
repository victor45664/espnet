from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from models import Generator
from pathlib import Path
h = None
device = None
MAX_WAV_VALUE = 32768.0

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    # print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    # print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    input_dir=Path(a.path_input_mels)
    mel_files = list(input_dir.rglob('*.npy'))


    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for mel_file in mel_files:
            mel_file_path=str(mel_file)
            x = np.load(mel_file_path)
            x = torch.FloatTensor(x).transpose(1,0).unsqueeze(0).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            write(mel_file_path[:-4]+".wav", h.sampling_rate, audio)



def main():
    # print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input_mels', default='mel_target')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config_file', required=True)
    a = parser.parse_args()

    config_file = a.config_file
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

