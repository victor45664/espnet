#!/usr/bin/env python

import numpy as np
import sys
import kaldi_io
import os

ppg_scp=sys.argv[1]
ppg_save_dir=sys.argv[2]

#if not os.path.exists(ppg_save_dir):
#    os.makedirs(ppg_save_dir)

for key,mat in kaldi_io.read_mat_scp(ppg_scp):
  fname=os.path.join(ppg_save_dir, key+'.ppg.npy')
  np.save(fname, mat, allow_pickle=False)
