# CS570

![](image.png)

## 0. Requirements

CUDA version : 11.8

Clone this repository and run the below line.

```
pip install -r requirements.txt
```


## 1. Download the dataset.

```
cd datasets
python kth_download.py

mkdir utdmad
cd utdmad
wget http://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip
unzip RGB.zip

cd ..
```

Make the downloaded file in the folder kth. As ./kth/boxing/person01_boxing_d1_uncomp.avi.
For the UTD-MHAD dataset, we have to set the data folder as cs570/datasets/utdmad/RGB/a1_s1_t1_color.avi.

## 2. Run the model

### Baseline model

with 5 channels: Gray frames, grad-x frames, grad-y frames, opt-x frames, opt-y frames

```
python run.py --lr [learning rate] --num_epochs [number of epochs]
python run_utdmad.py --lr [learning rate] --num_epochs [number of epochs]
```

### FFT model

with 2 channels: Amplitude frames, phase frames

You can choose <code>cut_param</code> from 0.7, 0.8, and 1.0(no cutting).

```
python run.py --fft FFT --cut_param [cutting ratio] \
              --dataset_dir [dataset directory] --lr [learning rate] \
              --num_epochs [number of epochs]
python run_utdmad.py --fft FFT --cut_param [cutting ratio] \
                     --dataset_dir [dataset directory] --lr [learning rate] \
                     --num_epochs [number of epochs]
```

### FFT3 model

with 3 channels: x-y phase, x-t phase, and y-t phase frames

You can choose <code>cut_param</code> from 0.7, 0.8, and 1.0(no cutting).

```
python run.py --fft FFT3 --cut_param [cutting ratio] \
              --dataset_dir [dataset directory] --lr [learning rate] \
              --num_epochs [number of epochs]

python run_utdmad.py --fft FFT3 --cut_param [cutting ratio] \
                     --dataset_dir [dataset directory] --lr [learning rate] \
                     --num_epochs [number of epochs]
```




### KTH Dataset

@inproceedings{inproceedings,
author = {Roth, Peter M. and Mauthner, Thomas and Khan, Inayatullah and Bischof, Horst},
year = {2009},
month = {11},
pages = {546 - 553},
title = {Efficient human action recognition by cascaded linear classifcation},
doi = {10.1109/ICCVW.2009.5457655}
}

### UTD-MHAD Dataset
https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
