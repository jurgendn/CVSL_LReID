#!/bin/bash
export PYTHONPATH=.
python3 tools/train.py --main-config configs/main_config.yaml \
--shape-config configs/shape_embedding.yaml \
--device cpu \
--num-workers 0