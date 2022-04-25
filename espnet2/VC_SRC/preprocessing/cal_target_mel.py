

import sys
sys.path.append('')

from espnet2.VC_SRC import melspectrogram,load_wav
import numpy as np
import os
import pandas as pd




def cal_mel_target(data_dir):
    wav_scp_path=os.path.join(data_dir,"wav.scp")
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
        wav = load_wav(wav_path)
        mel_spectrogram = melspectrogram(wav).astype(np.float32)  # 这里提取MFCC特征
        np.save(mel_target_path, mel_spectrogram.T, allow_pickle=False)
        mel_scp.writelines(utt_id+" "+mel_target_path+"\n")





if __name__ == '__main__':
    import sys
    data_dir=sys.argv[1]
    #data_dir="$root_path/tf_project/a2m_VC/models/baseline/data/train"

    cal_mel_target(data_dir)











