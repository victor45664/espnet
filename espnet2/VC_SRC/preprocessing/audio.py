
import librosa
import librosa.filters

import numpy as np
from scipy import signal
from espnet2.VC_SRC.preprocessing.default_hparams import hparams_for_preprocessing as hparams

def load_wav(path):
  return librosa.core.load(path, sr=hparams.target_sample_rate)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), 24000)

def preemphasis(x):
  return signal.lfilter([1, -0.97], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -0.97], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - 20
  return _normalize(S)


def spectrogram_rms(y):
  D = _stft(preemphasis(y))
  S, phase = librosa.magphase(D)
  rms = librosa.feature.rms(S=S)
  S = _amp_to_db(np.abs(D)) - 20
  return _normalize(S).astype(np.float32), rms.astype(np.float32)





def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(24000 * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)





def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)





def _stft_parameters():
  n_fft = (1025 - 1) * 2
  hop_length = int(10/ 1000 * 24000)
  win_length = int(50 / 1000 * 24000)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (1025 - 1) * 2
  return librosa.filters.mel(24000, n_fft, n_mels=80, fmin=40)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _normalize(S):
  return np.clip((S - -100) / --100, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * --100) + -100


# f0 and vvu feature extract:
# def world_lf0_extract(wav_path):
#   wav, fs = sf.read(wav_path)
#   if fs != 16000:
#     raise ValueError('input sample rate of durian4vc must be 16000')
#   f0, timeaxis = pyworld.harvest(wav, fs, frame_period=10)
#   aperiodicity = pyworld.d4c(wav, f0, timeaxis, fs)
#
#   f0 = f0[:, None]
#   lf0 = f0.copy()
#   nonzero_indices = np.nonzero(f0)
#   lf0[nonzero_indices] = np.log(f0[nonzero_indices])
#
#   vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
#
#   return lf0, vuv

