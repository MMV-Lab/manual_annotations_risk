# manual_annotations_risk

This repository provides the code to reproduce our results published in [??].

## Setup

We use Linux for all our processing and so this repository is based on the Linux-Shell. But with a little tweaking, all code should run under Windows and Mac as well. Setting up all environments is beyond this repo, we refer to the respective repository each ([EmbedSeg MMV_Im2Im version](compare_predictions.sh), [StarDist-3D](compare_predictions.sh), [Cellpose](compare_predictions.sh)). You will need an environment each, make sure you activate the right environment before training models or running the inference.

## Download the data

To download the DNA-dye and Lamin B1 image data published with [], run `python get_nuclei_data.py`. To download provided annotated masks and trained models from Zenodo, just run `get_masks_models.py`.

## Training


To follow Cellpose and EmbedSeg API, we need to slightly reorganize all the image files for training, so you will have duplicated files in each folder, which can be removed after training. But all reorganizing steps are covered by our training scripts.
To train nuclei instance segmentation models, go to the corresponding folder and run train.sh for each Cellpose and EmbedSeg and train.py for StarDist-3D. Per default, Napari-GT masks are used as training data. But you can simply change this.

## Inference

To get the predictions for Cellpose or StarDist-3D, run the predict.py Python script in the respective folder. To get EmbedSeg results, you need to run predict.sh .

## Evaluation

To evaluate the manual annotations, go to Evaluation => annotations and run compare_annotations.sh . This will automatically compare the volumes of each Napari-GT and Slicer-GT with bioGT.
For the predictions, go to Evaluation => predictions and run compare_predictions.sh . Per default, StarDist-3D's predictions are analyzed. You can simply adjust the method variable in each Python script to analyze the other results.