

import os
import numpy as np
from espnet2.VC_SRC.evaluation.vocoder import None_gen




def result_to_dir(keys,mels,mels_length,dir,vocoder=None_gen):
    assert len(keys)==mels.shape[0]

    if not os.path.exists(dir):
        os.makedirs(dir)



    for i in range(len(keys)):
        key=keys[i]
        length=mels_length[i]
        mel_save_path=os.path.join(dir,key+".npy")
        wav_save_path=os.path.join(dir,key+".wav")
        np.save(mel_save_path,mels[i,0:length,:])
        vocoder(mel_save_path,wav_save_path)
















