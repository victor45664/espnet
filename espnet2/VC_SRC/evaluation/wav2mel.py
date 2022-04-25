

import sys
sys.path.append('')

from espnet2.VC_SRC import melspectrogram,load_wav
import numpy as np
import os


def cal_mel_target(dir):    #这个函数遍历文件夹中所有wav文件并且计算相应的mel谱，保存为同名.npy
    for root, dirs, files in os.walk(dir):
        for f in files:
            if ".wav" in f:
                wav_path=os.path.join(root, f)
                print(wav_path)
                wav = load_wav(wav_path)
                mel_spectrogram = melspectrogram(wav).astype(np.float32)  # 这里提取MFCC特征
                np.save(wav_path[:-4]+'.npy', mel_spectrogram.T, allow_pickle=False)






if __name__ == '__main__':
    import sys
    data_dir=sys.argv[1]
    #data_dir="$root_path/tf_project/a2m_VC/models/baseline/data/train"

    cal_mel_target(data_dir)











