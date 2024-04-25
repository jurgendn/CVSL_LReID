#!/bin/bash
export PYTHONPATH=.
python tools/get_orientation.py --config configs/hrnet.yaml \
                                --pretrained pretrained/model_hboe.pth \
                                --dataset /Users/jurgendn/Documents/projects/personal/dataset/LTCC_ReID/ \
                                --target-set query
