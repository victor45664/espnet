from librosa.filters import mel as librosa_mel_fn
import math
import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import librosa

global mel_basis, hann_window
mel_basis = {}
hann_window = {}
def load_wav_hifigan(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):



    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



def hifigan_mel(wav_path):
    n_fft = 2048
    num_mels = 80
    sampling_rate = 24000
    hop_size = 240
    win_size = 1200
    fmin = 40
    fmax = 8000

    # load wav
    filename = wav_path
    MAX_WAV_VALUE = 32768.0

    audio=librosa.core.load(wav_path, sr=sampling_rate)[0]


    #audio = normalize(audio) * 0.95 # librosa.util.normalize, used only for training
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0) # this is the input for mel_spectrogram()

    # compute mel
    # parameter values are taken from https://github.com/jik876/hifi-gan/blob/master/config_v3.json. Only sampling_rate is modified to suit the sample rate of the loaded wav.
    y = audio


    #segment_size = 8192  Not used in mel_spectrogram. Before training, audio of this length will be randomly sampled and used for training
    # num_freq = 1025 Doesn't seem to be used during training
    hifigan_mel= mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
    hifigan_mel = hifigan_mel.squeeze() # this is passed into the generator

    return hifigan_mel


def cal_mel_target(data_dir):
    wav_scp_path=os.path.join(data_dir,"wav_mel.scp")
    mel_scp_path=os.path.join(data_dir,"mel_target.scp")
    mel_dir_path=os.path.join(data_dir,"mel_target")

    if not os.path.exists(mel_dir_path):
        os.mkdir(mel_dir_path)

    mel_scp=open(mel_scp_path,"w")

    wav_scp = pd.read_csv(wav_scp_path, header=None, delim_whitespace=True, names=['key', 'wav_path'])
    for i in range(len(wav_scp)):
        utt_id=wav_scp['key'][i]
        wav_path=wav_scp['wav_path'][i]
        mel_target_path=os.path.join(mel_dir_path,utt_id+".npy")
        mel = hifigan_mel(wav_path)
        np.save(mel_target_path, mel.T, allow_pickle=False)
        mel_scp.writelines(utt_id+" "+mel_target_path+"\n")


if __name__ == '__main__':


    import sys
    data_dir=sys.argv[1]
    #data_dir="$root_path/tf_project/a2m_VC/models/baseline/data/train"

    cal_mel_target(data_dir)


