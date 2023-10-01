#!/bin/bash

python get_slices.py
python -m cellpose --train --dir ./train_data/Napari-GT --pretrained_model nuclei --chan 0 --chan2 0 --n_epochs 1000 --min_train_masks 0 --verbose --use_gpu