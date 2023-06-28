# MQU ML Project 1
Mariia Eremina, Qiming Sun, Ulysse Widmer

https://www.aicrowd.com/challenges/epfl-machine-learning-higgs

## Setup
Clone this repo, download the dataset available at the link above and put the `.csv` files in the root folder (same place as `run.ipynb`.
To get the code working, at the root folder you should have at least:
 - `helpers.py` contains helpers functions to manipulate csv, preprocess data and various other functions
 - `implementations.py` contains the 6 models implementations
 - `run.ipynb` contains the code to load, explore, preprocess, score the data and create submissions
 - `train.csv` the labeled data used for training
 - `test.csv` the data to be labeled
 
## Runing the models
The file `run.ipynb` contains a comprehensive display of the models. You can simply load the root folder in a jupyter notebook and run the `run.ipynb` file to see our results. The file will automatically create a submission csv file for each model.

## Results
The predictions are printed in the `*_submission.csv` files. Our highest score on AICrowd was the result of the Least Squares predictions (`ls_submission.csv`).
