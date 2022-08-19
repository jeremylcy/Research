# Mask-RCNN for Vehicle Defects Detection

## Intro

Detects vehicle defects via feeding images of defects (currently only scratches or dents) and output would the same image except the defected parts would be marked out.
Ran via RTX2080 GPU on Windows

## Technologies

Project is created with:
- Python = 3.7.X
- Tensorflow = 2.4.0
- Cuda Toolkit = 11.2
- Cudnn = 8.1

## Setup

```bash
conda create -n Vehicle_Defects python=3.7
conda activate Vehicle_Defects
```
1. Get the folder from the aison hard disk/drive and copy the "Vehicle_Defects" folder to your computer
   (If folder not found, ask someone in the office on how to get it)
2. Go to that folder through your prompt, example "cd /path/to/Vehicle_Defects"
3. Enter "pip install -r requirements.txt"

If the above does not work, try the below following:
- For Windows
```bash
conda env create -f Windows_environment.yml
conda activate Vehicle_Defects
```
- For Ubuntu/Linux
```bash
conda env create -f Ubuntu_environment.yml
conda activate Vehicle_Defects
```

## Dataset preparation

```bash
  - Vehicle_Defects
       - mrcnn
       - logs
       - my_dataset
            - train
                 - 1.jpg
                 - 2.jpg
                 - data.json
            - val
                 - 1.jpg
                 - 2.jpg
                 - data.json
```

## Usage

- Training

Training of custom weights (Using Pre-trained weights)
```bash
    python BG+1.py train --dataset=/path/to/dataset --weights=coco
```
Training of custom weights (Using Current or Existing trained weights)
```bash
    python BG+1.py train --dataset=/path/to/dataset --weights=/path/to/trained_model.h5
```

- Evaluation

For trained weights evaluation (Using Current or Existing trained weights)
```bash
    python BG+1.py evaluate --dataset=/path/to/dataset --weights=/path/to/model.h5
```

- Prediction

For images folder prediction
```bash
    python BG+1.py splash --weights=/path/to/model.h5 --image_path=/path/to/images_folder
```
For webcam prediction
```bash
    python BG+1.py splash --weights=/path/to/model.h5 --video_path=0
```
For video folder prediction (not fixed yet)
```bash
    python BG+1.py splash --weights=/path/to/model.h5 --video_path=/path/to/video_folder
```