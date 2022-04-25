#!/bin/bash
#victor 2020.9.12

mutation=$1

$root_path/anaconda3/envs/pytorch/bin/python -u $(dirname $0)/eval.py $mutation 0 || exit 1;


