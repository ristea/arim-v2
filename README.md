#  Automotive Radar Interference Mitigation data sets (ARIM-v2)                                                                                    

We propose a novel large scale database consisting of radar data samples, generated automatically while trying to replicate a realistic automotive scenario with variable interference sources.
  
We provide two ways to obtain data:
- directly by downloading the data from below listed links
- generate the data by using the provided scripts

-----------------------------------------

![map](resources/example.jpg)

-----------------------------------------                                                                                                                                      
## Download data set

##### You can download the data set from here: 

https://fmiunibuc-my.sharepoint.com/:f:/g/personal/radu_ionescu_fmi_unibuc_ro/ErpEnoVjRcNAqx-pKxYelDABFnnWQ1HRVJZWFHbMtWc4ZQ?e=YYDMIq

##### You can get the data set paper from here:
http://arxiv.org/abs/2008.05948

## Generate data set
#### In order to generate the ARIM-v2 data set:
1. Run the matlab script arim_matlab/main.m
2. Move the generated file (arim1.mat) in X directory
3. Run the matlab script arim2_matlab/main.m 
4. Run again the matlab script arim2_matlab/main.m, but modify the **nr_interferences** variable to 3.
5. Move the generated files (arim2.mat and arim3.mat) in X directory
3. Run the process.py script as follows:
```bash
python process.py --arim_data_path path/to/X/dir --output_dataset_path path/to/save
```

#### Information

After the above steps you will have in the path/to/save directory two files: **arim-v2_train.npy** and **arim-v2_test.npy**.
Those files contains the subsets for training (which could be split also in train and evaluation, as described in our paper) and testing.

In order to load the data in python you should run:
```python
import numpy as np
arim = np.load("path/to/dataset", allow_pickle=True)

sb_raw = arim[()]['sb'] # Data with interference
sb0_raw = arim[()]['sb0'] # Data without interference
amplitudes = arim[()]['amplitudes'] # Amplitude information for targets
```
> In order to work properly you need to have a python version older than 3.6
>> We used the following versions:
>> python 3.6.8,
>> numpy 1.17.3

## Run our pretrained models

In **trained_models** there are ours pretrained models. In order to perform inference you have to use **main.py** script.
In addition, we added a config file **config.json** with parameters and some paths. You must adapt this configuration file.
  
```bash
python main.py
```

## Cite us
```
@article{ristea2020estimating,
    title={Estimating Magnitude and Phase of Automotive Radar Signals under Multiple Interference Sources with Fully Convolutional Networks},
    author={Nicolae-Cătălin Ristea and Andrei Anghel and Radu Tudor Ionescu},
    journal={arXiv preprint arXiv:2008.05948},
    year={2020}
}
```
## You can send your questions or suggestions to: 
r.catalin196@yahoo.ro, raducu.ionescu@gmail.com

### Last Update:
August 14, 2020 



