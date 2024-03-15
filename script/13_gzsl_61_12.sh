#!/bin/bash
# --pretrain_epoch 0 means pretrain the first epoch
CUDA_VISIBLE_DEVICES=0 python ./model/main.py \
--dataset RSSDIVCS --pretrain_epoch 4 --pro_type 4 --pro_thr 1.0 \
--additional_loss \
--calibrated_stacking 0.7 \
--cuda --nepoch 30 --batch_size 64 --train_id 123 \
--pretrain_lr 1e-4 --classifier_lr 1e-6 \
--xe 1 --attri 1e-2 --regular 0 \
--l_xe 1 --l_attri 8e-2  --l_regular 0 \
--xe_pro 1 \
--cpt 1e-9 --use_group \
--train_mode 'distributed' --n_batch 300 --ways 8 --shots 3 \
--image_embedding_path  /home/xwj/data/2023_code/RS_ZSL/ISPRS_2023_clean/RSSDIVCS/files/61_images.mat \
--class_embedding_path  /home/xwj/data/2023_code/RS_ZSL/ISPRS_2023_clean/RSSDIVCS/files/ann_attr_finetune_61_splits.mat \
--sel_res layer \
--gzsl --calibrated_stacking 0.0001 --pro_start_epoch -1



