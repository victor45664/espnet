#!/bin/bash
#victor 2020.9.12

mutation=$1

python -u $(dirname $0)/eval.py $mutation 0 || exit 1;
