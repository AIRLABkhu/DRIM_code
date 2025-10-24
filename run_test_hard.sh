#!/bin/bash

conda activate mono

python -u test_hard.py --exp /data2/local_datasets/DRIM/result/test --encoder swinv2 --decoder upernet --model_size tiny --batch_size 1 --seed 42 --save_image