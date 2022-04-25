import os
import librosa
import librosa.filters
import pyworld
import kaldi_io
import subprocess
import soundfile as sf
import numpy as np

from scipy import signal
#from VC_SRC.preprocessing.default_hparams import hparams_for_preprocessing as hparams

from espnet2.VC_SRC import hparams_for_preprocessing as hparams

def load_wav(path):

  return librosa.core.load(path, sr=hparams.target_sample_rate)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), hparams.target_sample_rate)

def preemphasis(x):
  return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def spectrogram_rms(y):
  D = _stft(preemphasis(y))
  S, phase = librosa.magphase(D)
  rms = librosa.feature.rms(S=S)
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S).astype(np.float32), rms.astype(np.float32)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.target_sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.target_sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.target_sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.target_sample_rate, n_fft, n_mels=hparams.num_mels, fmin=hparams.fmin)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

# f0 and vvu feature extract:
def world_lf0_extract(wav_path):
  wav, fs = sf.read(wav_path)
  if fs != hparams.source_sample_rate:
    raise ValueError('input sample rate of durian4vc must be 16000')
  f0, timeaxis = pyworld.harvest(wav, fs, frame_period=hparams.frame_shift_ms)
  aperiodicity = pyworld.d4c(wav, f0, timeaxis, fs)

  f0 = f0[:, None]
  lf0 = f0.copy()
  nonzero_indices = np.nonzero(f0)
  lf0[nonzero_indices] = np.log(f0[nonzero_indices])

  vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]

  return lf0, vuv


def kaldi_lf0_extract(wav_path):
  uttid = os.path.basename(wav_path)[:-4]
  # TO DO: Speed up this
  # current procedule is not optimal
  # as it repeatly open and close ark file
  kaldi_piped="ark:compute-kaldi-pitch-feats --max-f0=600 --sample-frequency={sf} scp:'echo {utt} {wavfile} |' ark,t:- 2>/dev/null |".format(sf=str(hparams.source_sample_rate), wavfile=wav_path, utt=uttid)
  for key,mat in kaldi_io.read_mat_ark(kaldi_piped):
    pov_f0 = mat
    lf0 = np.log(pov_f0[:,1])
    lf0 = lf0.reshape(len(lf0),1)
    nccf = pov_f0[:,0]
    nccf = nccf.reshape(len(nccf),1)
    #print(pov_f0.shape)
  return lf0, nccf


def execute_command(command):
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode is not 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))

def ppg_extract(wav_path,tmpdir):
  uttid = os.path.basename(wav_path)[:-4]
  
  execute_command("""mkdir -p tmp/ 
                    source kaldi_path.sh
		    paste-feats --length-tolerance=2 \
                    "ark:compute-mfcc-feats --config=$KALDI_CONF/mfcc_hires.conf \
                    scp,p:'echo {utt} {wavfile} |' ark:- |"    \
                    "ark,s,cs:compute-kaldi-pitch-feats  --config=$KALDI_CONF/pitch.conf \
                    scp,p:'echo {utt} {wavfile} |' ark:- |    \
                    process-kaldi-pitch-feats  ark:- ark:- |" ark:{tmp}/feats.{utt}.ark  2>/dev/null""".format(wavfile=wav_path, utt=uttid, tmp=tmpdir))
  execute_command("""compute-cmvn-stats --spk2utt=ark:'echo {utt} {utt} |' ark:{tmp}/feats.{utt}.ark ark:{tmp}/cmvn.{utt}.ark 2>/dev/null """.format(utt=uttid, tmp=tmpdir))
  execute_command("""ivector-extract-online2 --config=$KALDI_CONF/ivector_extractor.conf ark:'echo {utt} {utt} |' 'ark:select-feats 0-39 ark:{tmp}/feats.{utt}.ark ark:- |' ark:{tmp}/ivector.{utt}.ark 2>/dev/null """.format(utt=uttid, tmp=tmpdir))
  kaldi_chain_piped="""ark:nnet3-chain-compute-post \
                    --use-gpu=no --online-ivector-period=10 \
                    --online-ivectors=ark:{tmp}/ivector.{utt}.ark\
                    --transform-mat=$KALDI_CONF/transform.mat \
                    $KALDI_CONF/final.raw $KALDI_CONF/den.fst \
                    ark:{tmp}/feats.{utt}.ark \
                    ark:- 2>/dev/null |""".format(utt=uttid, tmp=tmpdir)
  kaldi_nnet3_piped="""ark:nnet3-compute \
                       --use-gpu=no --online-ivector-period=10 \
                       --online-ivectors=ark:{tmp}/ivector.{utt}.ark --apply-exp=true \
                       $KALDI_CONF/final.mdl "ark,s,cs:apply-cmvn --norm-means=true --norm-vars=true  --utt2spk=ark:'echo {utt} {utt} |' ark:{tmp}/cmvn.{utt}.ark ark:{tmp}/feats.{utt}.ark ark:- |" ark:-  2>/dev/null | transform-feats $KALDI_CONF/transform.mat ark:- ark:- 2>/dev/null |""".format(utt=uttid, tmp=tmpdir)
  model="nnet3-copy --nnet-config=$KALDI_CONF/bottle_neck.config $KALDI_CONF/final.mdl -|"
  kaldi_bottleneck_piped="""ark:nnet3-compute \
                       --use-gpu=no --online-ivector-period=10 \
                       --online-ivectors=ark:{tmp}/ivector.{utt}.ark \
                       "{model}" "ark,s,cs:apply-cmvn --norm-means=true --norm-vars=true  --utt2spk=ark:'echo {utt} {utt} |' ark:{tmp}/cmvn.{utt}.ark ark:{tmp}/feats.{utt}.ark ark:- |" ark:-  2>/dev/null |""".format(utt=uttid, model=model, tmp=tmpdir) 
  # print(kaldi_piped)
  # mat=kaldi_io.read_mat_ark(uttid+".ppg.mat")
  #  for key,mat in kaldi_io.read_mat_ark("ark:{tmp}/ppg.{utt}.ark".format(utt=uttid, tmp=tmpdir)):
  for key,mat in kaldi_io.read_mat_ark(kaldi_bottleneck_piped):
      ppg_mat = mat  
  # if os.path.exists(tmpdir+'/'+feats.{utt}.ark'.format(utt=uttid)):
  #     os.remove(tmpdir+'/'+'feats.{utt}.ark'.format(utt=uttid))
  return ppg_mat
  
  # return mat
