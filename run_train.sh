#!/bin/bash

conda activate mono

python -u train.py --exp /data2/local_datasets/DRIM/result/test --seed 42 --encoder swinv2 --decoder upernet --model_size tiny \
                   --crop --warmup_epoch 0 --epoch 100 \
		   --encoder_lr 5e-4 --lr 5e-4 --weight_decay 1e-4 --decay_factor 0.9 --t_max 10 --batch_size 32 --eval_batch_size 16 \
                   --save_code --use_amp --val_frequency 1 --interval 100 #--save_image