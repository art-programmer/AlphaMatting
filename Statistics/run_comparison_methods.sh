#!/usr/bin/env zsh

exe=/home/erik/Projects/ParallelFusion/build/AlphaMatting/AlphaMatting
data_set=GT24

data_file=raw-time-v-energy.csv


${exe} -img_name=${data_set} -num_threads=1 -num_proposals=1 -exchange_amount=0 -exchange_interval=1
cp $data_file fusion_move.csv

${exe} -img_name=${data_set} -num_threads=4 -num_proposals=1 -exchange_amount=0 -exchange_interval=1
cp $data_file parallel_fusion_move.csv

${exe} -img_name=${data_set} -num_threads=4 -num_proposals=1 -exchange_amount=1 -exchange_interval=5
cp $data_file ss-mf.csv

${exe} -img_name=${data_set} -num_threads=4 -num_proposals=3 -exchange_amount=0 -exchange_interval=1
cp $data_file sf-ss.csv

${exe} -img_name=${data_set} -num_threads=4 -num_proposals=3 -exchange_amount=3 -exchange_interval=5
cp $data_file sf.csv

${exe} -img_name=${data_set} -num_threads=4 -num_proposals=3 -exchange_amount=2 -exchange_interval=5
cp $data_file bopt.csv