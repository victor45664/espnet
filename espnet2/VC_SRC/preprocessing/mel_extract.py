import os
import sys
import shutil
from glob import glob
from pathlib import Path
import time
import multiprocessing as mp
import numpy as np

import torch
import torchaudio

#if len(sys.argv)!=10:
#    print("Usage: ")
#    print("python torch_feature_extractor.py <path_to_wav_dir> <path_to_feat_dir> <sampling_rate> <fft_len> <win_len> <hop_len> <f_min> <f_max> <n_mels>")
#    sys.exit(1)

# input audio directory
wav_dir = sys.argv[1]

# Output features directory
out_dir = sys.argv[2]

# sampling frequency
fs = int(sys.argv[3]) # 16000

# fft length
fft_len = int(sys.argv[4]) # 1024

# window length
win_len = int(sys.argv[5]) # 50ms X 16000 = 800

# frame shift
hop_len = int(sys.argv[6]) # 12.5ms X 16000 = 200

# min frequency
f_min = float(sys.argv[7]) # 0

# max frequency
f_max = int(sys.argv[8]) # 8000

# number of mels
n_mels = int(sys.argv[9]) # 80

mfb_dir  = os.path.join(out_dir, 'mfb' )

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(mfb_dir):
    os.mkdir(mfb_dir)

def get_wav_filelist(wav_dirs, mfb_dir):
    wav_files = []

    wav_paths = wav_dirs.split(',')
    for wav_dir in wav_paths:
#        for file in [y for x in os.walk(wav_dir) for y in glob(os.path.join(x[0], '*.wav'))]:
        for file in Path(wav_dir).rglob('*.wav'):
            file_id = os.path.basename(file).split(".")[0]
            mfb_file = os.path.join(mfb_dir, file_id + ".npy")

            if file and not os.path.exists(mfb_file):
                wav_files.append(file)

    wav_files.sort()

    return wav_files

def fbank_feature_extract(wav_file):
    """EXTRACT WORLD FEATURE VECTOR."""

    print("now processing %s " % (wav_file))
    file_name = os.path.basename(wav_file)
    file_id   = os.path.splitext(file_name)[0]
    mfb_file  = os.path.join(mfb_dir, file_id)

    spk_id = os.path.basename(os.path.dirname(os.path.dirname(wav_file)))
    #mfb_file = os.path.join(mfb_dir, spk_id + '-' + file_id)

    # load wavfile
    waveform, sample_rate = torchaudio.load(wav_file)
    if sample_rate != fs:
        print('resample waveform from {}Hz to {}Hz'.format(sample_rate, fs))
        waveform = torchaudio.transforms.Resample(sample_rate, fs)(waveform)

    mel_spec = melspec_ops(waveform)

    log_mel = torch.log10(mel_spec)
    log_mel = log_mel.transpose(1, 2)
    log_mel = np.squeeze(log_mel, axis=0)
    feat = log_mel.detach().numpy()
    feat[feat == -np.inf] = -8
    np.save(mfb_file, feat.astype(np.float32), allow_pickle=False)


print("--- Feature extraction started ---")
start_time = time.time()

# get wav files list
wav_files = get_wav_filelist(wav_dir, mfb_dir)

melspec_ops = torchaudio.transforms.MelSpectrogram(sample_rate=fs,
                 n_fft=fft_len,
                 win_length=win_len,
                 hop_length=hop_len,
                 f_min=f_min,
                 f_max=f_max,
                 n_mels=n_mels)


# do multi-processing
pool = mp.Pool(mp.cpu_count())
pool.map(fbank_feature_extract, wav_files)

# DEBUG:
#for nxf in xrange(len(wav_files)):
#    process(wav_files[nxf])

# clean temporal files
#shutil.rmtree(sp_dir, ignore_errors=True)
#shutil.rmtree(f0_ori_dir, ignore_errors=True)


print("You should have your features ready in: "+ out_dir)

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- Feature extraction completion time: %d min. %d sec ---" % (m, s)))

