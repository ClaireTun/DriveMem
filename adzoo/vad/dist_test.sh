#!/usr/bin/env bash

CONFIG="/disk1/zhaoxue/bev/RL/Bench2Drive/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d_VCA_mem_DSU.py"
CHECKPOINT="/disk1/zhaoxue/bev/RL/Bench2Drive/Bench2DriveZoo/work_dirs/VAD_base_e2e_b2d_VCA_mem_DSU/epoch_6.pth"
GPUS=1
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
