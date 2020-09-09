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


## Usage
Run following to preprocess the dataset

```bash
python -m data_pipeline.preprocessing
python -m data_pipeline.preprocessing_bis
python -m data_pipeline.preprocessing_ter
```

Once you have data preprocessed run `train.py`.
```bash
python train.py
```
Or for resumable training run 
```bash
python resumable.py
```
## Dataset
This implementation uses The ICDAR 2019 dataset.  
For more information and download sources check [Niansong Zhang's work](https://github.com/zzzDavid/ICDAR-2019-SROIE)
## License
[GPLV3](https://choosealicense.com/licenses/gpl-3.0/)