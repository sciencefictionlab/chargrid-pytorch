# Chargrid Pytorch

This is the pytorch implementation of Chargrid paper [Arxive Link](https://arxiv.org/abs/1809.08799).  

Pre-processing has been taken from Antoine Delplace's repository.  
Also, for tensorflow2.0 implementation check out his repository [Link Here](https://github.com/antoinedelplace/Chargrid)

[See this for model description](https://github.com/sciencefictionlab/chargrid-pytorch/blob/master/Model%20Architecture.MD)

## Installation
After cloning this repository

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```
Copy `env.sample` to `.env`
Then add folder paths for the dataset in `.env` file

## Dataset
This implementation uses The ICDAR 2019 dataset.  
For more information and download sources check [Niansong Zhang's work](https://github.com/zzzDavid/ICDAR-2019-SROIE)

The default directories for data are: (can be changed from env file)  
image inputs in `./data/img_inputs/` (eg file1.jpg)  
ground truth classes in `./data/gt_classes/` (eg file1.json)  
ground truth ocr in `./data/gt_boxes/` (eg file1.txt)

## Data Preprocessing
Run following to preprocess the dataset

```bash
python -m data_pipeline.preprocessing
python -m data_pipeline.preprocessing_bis
python -m data_pipeline.preprocessing_ter
```

## Model Training
After data preprocessing, model can be trained with `train.py` script.
The arguments available in `train.py` are:
```
-r --restart 	To restart the training from epoch 0. (default will resume from checkpoint provided).
-c --checkpoint	Epoch number to load model from.
-e --epochs 	Number of epochs to run.
```
To train from start:
```bash
python train.py -r
```
To resume training from some checkpoint:
```bash
python train.py -c 9 
```

## Results

### SROIE
[To be updated]

## License
[GPLV3](https://choosealicense.com/licenses/gpl-3.0/)