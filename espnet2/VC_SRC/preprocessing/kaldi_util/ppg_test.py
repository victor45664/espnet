import argparse
import os
import numpy as np
import subprocess
import kaldi_io
from multiprocessing import cpu_count
from util import audio

def preprocess_run(args):
    input_wav = args.input
    output_file = args.output
    ouput = audio.ppg_extract(input_wav,"tmp")
    np.save(output_file, ouput, allow_pickle=False)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', default=os.path.expanduser('./output.ppg')) #默认输出为output.ppg.npy
    args = parser.parse_args()
    preprocess_run(args)


if __name__ == "__main__":
    main()