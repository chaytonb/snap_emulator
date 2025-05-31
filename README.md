# U-Net Emulator Repository
This repository contains the code for the U-Net emulator created for my second project. 

## Data Folder
The data contained in this GitHub respository is just a small subset of the full dataset which can be used to verify
that the code works correctly. The folder data contains a file 'sample_training_data.npz' which contains a set of 20
training examples. The files in this repository are currently set up to run from this sample subset, however this file 
can be replaced with the full data and run the same way. When opened, this data file contains two arrays, labeled 
'' and ''. 

'pred_array' contains the predictor variables which are fed as input to the U-Net model. The array dimensions are 
(sample, variable, width, height). The indices of the meteorological variables are as follows:

0: x-wind at time T
1: x-wind ant time T+6
2: y-wind at time T
3: y-wind at time T+6
4: atmospheric boundary layer height at time T
5: atmospheric boundary layer height at time T+6
6: surface air pressure at time T
7: surface air pressure at time T+6

'target_array' contains the accumulated concentration field at time T+6. It has dimensions (sample, width, height).

The full dataset after pre-processing and scaling is uploaded on a google drive, which can be accessed from the 
following link: 

Also contained within the data folder is an output file from the SNAP model, 'ringhals_20230101_00Z.nc'. This file is
only used in the code for retrieving the geospatial information required for plotting the data in the correct
projection.

Also important to note is that the concentration values produced by the emulator are adjusted by +20. For plotting 
purposes, this 20 is subtracted again so that the concentration is then in ln(c) where c is the concentration
in g/m^2.

## Models Folder
The models folder contains the files 'unet_emulator_full.pth' which contains the model weights obtained by training 
the U-Net on the full dataset. 'unet_emulator_sample.pth' contains the model weights from training on the sample
dataset.

## U-Net Files

### unet.py
This file contains the U-Net class used throughout this project. 

### model_eval.py
This file contains the various functions used to evaluate and plot the predictions of the emulator. This file by 
default can be run which will create the figures and provide metrics for the sample dataset.

### unet_train.ipynb
This notebook provides the training loop used in the project. The version provided here is set up to work on the 
small sample dataset provided in the data folder. This can be simply replaced with the full dataset if desired.

## Other Files
### environment.yml
Contains the packages and versions of the Conda environment used throughout the development of this project.

