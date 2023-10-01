#!/bin/bash

# set paths to data
raw_source_folder="../data/raw/dna/"
masks_source_folder="../data/masks/Napari-GT/"

# create folder for training data
target_folder="./train_data/"
mkdir -p "$target_folder"

# copy first 20 raw images to training data folder, add suffix "_IM"
for file in $(ls "$raw_source_folder" | sort | head -n 20); do
    filename=$(basename "$file")
    extension="${filename##*.}"
    filename_without_extension="${filename%.*}"
    cp "$raw_source_folder$file" "$target_folder/${filename_without_extension}_IM.$extension"
done

# copy first 20 masks to training data folder, add suffix "_GT"
for file in $(ls "$masks_source_folder" | sort | head -n 20); do
    filename=$(basename "$file")
    extension="${filename##*.}"
    filename_without_extension="${filename%.*}"
    cp "$masks_source_folder$file" "$target_folder/${filename_without_extension}_GT.$extension"
done

# make cache folder for pre-training
mkdir -p $"./cache"

#Trainings
run_im2im --config ./train_pT.yaml
run_im2im --config ./train_finetune.yaml