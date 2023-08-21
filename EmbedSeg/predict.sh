#!/bin/bash

mkdir -p "../predictions/EmbedSeg/Napari-GT"
mkdir -p "./holdout/"

source_folder="../data/raw/dna/"
target_folder="./holdout/"

for file in $(ls "$source_folder" | sort -r | head -n 18); do   # we use only the last 18 images for inference, see manuscript
    cp "$source_folder$file" "$target_folder$file"
done

# run inference
run_im2im --config ./predict.yaml