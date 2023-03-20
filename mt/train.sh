#!/bin/bash

touch nmt/DEBUG.log
fsync -d 10 nmt/DEBUG.log &

SL=en
TL=vi
PE=spd_centrality
TASK=${SL}2${TL}_${PE}
module load pytorch/1.1.0
python3 -m nmt --proto $TASK

