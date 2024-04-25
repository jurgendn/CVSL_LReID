#!/bin/bash
export PYTHONPATH=.
python tools/visual_ranklist.py \
    --config configs/main_conf.yaml \
    --model-path pretrained/model_hboe.pth \
    --device cpu 