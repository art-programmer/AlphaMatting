#!/usr/bin/env zsh

exe=/home/erik/Projects/ParallelFusion/build/AlphaMatting/AlphaMatting
data_set=GT24
timeout_time=480
resize_factor=-1.0

data_file=raw-time-v-energy.csv


echo "Running fusion move"
${exe} -img_name=${data_set} -num_threads=1 -num_proposals=1 -exchange_amount=0 -exchange_interval=1 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} fusion_move.csv

echo "Running parallel fusion move"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=1 -exchange_amount=0 -exchange_interval=1 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} parallel_fusion_move.csv

echo "Running SF-MF"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=1 -exchange_amount=1 -exchange_interval=3 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} sf-mf.csv

echo "Running SF-SS"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=3 -exchange_amount=0 -exchange_interval=1 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} sf-ss.csv

echo "Running SF"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=3 -exchange_amount=2 -exchange_interval=3 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} sf.csv

echo "Running SS-Baeysian"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=2 -exchange_amount=1 -exchange_interval=1 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} bopt.csv

echo "Running SF-Grid"
${exe} -img_name=${data_set} -num_threads=2 -num_proposals=1 -exchange_amount=1 -exchange_interval=4 -timeout=${timeout_time} -resize_factor=${resize_factor}
cp ${data_file} grid.csv