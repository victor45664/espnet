
import sys
sys.path.append('')
import numpy as np
import os
import pandas as pd
import kaldi_io

from espnet2.VC_SRC import melspectrogram,load_wav
import librosa

def load_wav(path):
  return librosa.core.load(path, sr=24000)[0]


def cal_mel_target(data_dir):
    wav_scp_path=os.path.join(data_dir,"wav.scp")
    ark_path=os.path.join(data_dir,"feats.ark")

    wav_scp = pd.read_csv(wav_scp_path, header=None, delim_whitespace=True, names=['key', 'wav_path'])
    with open(ark_path, 'wb') as f:
        for i in range(len(wav_scp)):
            utt_id = wav_scp['key'][i]
            wav_path = wav_scp['wav_path'][i]



            wav = load_wav(wav_path)
            mel_spectrogram = melspectrogram(wav).astype(np.float32)  # 这里提取MFCC特征
            kaldi_io.write_mat(f, mel_spectrogram.T, key=utt_id)

# def cal_mel_target(data_dir):
#     wav_scp_path=os.path.join(data_dir,"wav.scp")
#     ark_path=os.path.join(data_dir,"feats.ark")
#
#     wav_scp = pd.read_csv(wav_scp_path, names=['code'])
#     wav_scp[['key', 'wav_path']] = wav_scp["code"].str.split(" ", 1, expand=True)
#     temp_wav_path=os.path.join(data_dir, "temp.wav")
#     with open(ark_path, 'wb') as f:
#         for i in range(len(wav_scp)):
#             utt_id=wav_scp['key'][i]
#             wav_path=wav_scp['wav_path'][i]
#
#             cmd=wav_path[:-3]+temp_wav_path
#
#             os.system(cmd)
#
#             wav = load_wav(temp_wav_path)
#             mel_spectrogram = melspectrogram(wav).astype(np.float32)  # 这里提取MFCC特征
#             kaldi_io.write_mat(f, mel_spectrogram.T, key=utt_id)



if __name__ == '__main__':
    import sys


    data_dir=sys.argv[1]

    cal_mel_target(data_dir)

    # from subprocess import PIPE, run
    # import subprocess
    #
    #
    # cmd='sox --vol 0.221527020709 -t wav $root_path/Data/056020296.WAV -t wav -'
    # output =     subprocess.Popen(cmd)
    # wav = np.fromstring(output, dtype=np.float32)
    #
    # print(wav.shape)

    # cmd='King-ASR-166_056020292 sox --vol 0.221527020709 -t wav $root_path/Data/056020296.WAV -t wav - |'
    # f=os.popen(cmd)
    # wav=load_wav(f)










