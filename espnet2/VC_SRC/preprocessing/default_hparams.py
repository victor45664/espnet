from espnet2.VC_SRC.utils import HParams


# Default hyperparameters:
hparams_for_preprocessing =HParams(

  # Audio:
  ppg_dims=850,
  num_mels=80,
  num_freq=1025,
  target_sample_rate=24000,
  source_sample_rate=16000,
  frame_length_ms=50,
  frame_shift_ms=10,
  preemphasis=0.97,
  fmin=40,
  min_level_db=-100,
  ref_level_db=20,


)







