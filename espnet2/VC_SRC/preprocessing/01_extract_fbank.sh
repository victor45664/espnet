#!/bin/bash


wav_dir=$1
out_dir=$2

python torch_feature_extractor.py ${wav_dir} ${out_dir} 16000 1024 800 200 0 8000 80
