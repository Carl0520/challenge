# Interspeech 2019 Baby Sound Challenge
This repository is a example for interspeech 2019 challenge work.

# Dependencies
Pytorhc v.1.0.0


python 3.6

# Data path structure

├── features_path 
│   ├── ComParE.train.csv
│   ├── ComParE.devel.csv
│   └── ComParE.test.csv
│   ├── auDeep-fused.train.csv
│   ├── auDeep-fusedE.devel.csv
│   └── auDeep-fused.test.csv
│   ├── IS10.train.csv
│   ├── IS10.devel.csv
│   └── IS10.test.csv
├── label
│   └── labels.csv

    

# Usage
1. You need to download the dataset first, and extract the feature and label into csv form.
2. run main.py first to pretrain the decoder.
3. run svm.py to evaluate.

# Result
![](result.png?raw=true)
