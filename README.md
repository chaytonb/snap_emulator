# U-Net Emulator Repository
This repository contains the code for the U-Net emulator created for my second project. The full dataset which I used
in this project was much too large to host on GitHub, so I have tried my best to make the code testable and the results
reproduceable. The repository is set up by default to run on a very small subset (30 training examples) and 10 epochs to verify that the code works,
however model produced from this sample dataset will of course perform very poorly compared to the full dataset. The sample
dataset should just take a few minutes to train on. The number of epochs can be reduced to reduce training time further.

Below is information about each file in the repository as well as instructions on how to train and use the model,
and how to run it using the full dataset if desired.

## Contents

### Data Folder
The data contained in this GitHub respository is just a small subset of the full dataset which can be used to verify
that the code works correctly. The folder data contains a file 'sample_training_data.npz' which contains a set of 30
training examples. The files in this repository are currently set up to run from this sample subset, however this file 
can be replaced with the full data and run the same way. When opened, this data file contains two arrays, labeled 
'pred_vars' and 'target_vars'. 

'pred_vars' contains the predictor variables which are fed as input to the U-Net model. The array dimensions are 
(sample, variable, width, height). The indices of the meteorological variables are as follows:

- 0: x-wind at time T
- 1: x-wind ant time T+6
- 2: y-wind at time T
- 3: y-wind at time T+6
- 4: atmospheric boundary layer height at time T
- 5: atmospheric boundary layer height at time T+6
- 6: surface air pressure at time T
- 7: surface air pressure at time T+6

'target_vars' contains the accumulated concentration field at time T+6. It has dimensions (sample, width, height).

Also contained within the data folder is an output file from the SNAP model, 'ringhals_20230101_00Z.nc'. This file is
only used in the code for retrieving the geospatial information required for plotting the data in the correct
projection.

Also important to note is that the concentration values produced by the emulator are adjusted by +20. In the plotting 
functions, this 20 is subtracted again so that the concentration is then in ln(c) where c is the concentration
in g/m^2.

### Models Folder
The models folder contains the files 'unet_emulator_report_version.pth' which contains the model weights obtained by training 
the U-Net on the full dataset, and is the set of weights which was used to produce the results used in the report. 
'unet_emulator_sample.pth' contains the model weights from training on the sample dataset.

### Report Folder
This folder contains a PDF of the project report, as well as the report_figures folder which contains the figures used in the report.

### config.py
This file controls the dataset used during the training and evaluation. This file must be edited when switching
between either using the sample dataset or the full dataset.

### environment.yml
Contains the packages and versions of the Conda environment used throughout the development of this project.

### model_eval.py
This file contains the various functions used to evaluate and plot the predictions of the emulator. This file by 
can be run to produce the various figures and metrics used in this report.

### unet.py
This file contains the U-Net class used throughout this project. 

### plot_dmomain.ipynb
Creates the domain map (Figure 2 in the report)

### unet_train.ipynb
This notebook provides the training loop for the U-Net used in the project. It is by default set up to run with 10 epochs as
an example, however it was run with 100 epochs for the actual project.

### report_figures.ipynb
This notebook is provided to recreate the figures which appear in the report. It is to be used in conjunction with the unet_emulator_report_version.pth weights/biases
and the test_set_report.npz test dataset.

## How to Run
By default, the code is set up to run on the sample dataset already in the repository. The unet_train.ipynb can be run
to train the U-Net. From this the model weights are saved in the models folder. Then, the model_eval.py file can be run
which produces the figures used in the report, and prints the mean IoU value of the predictions. These figures will be
placed in the figures folder and overwrite any existing contents.

The full dataset after all pre-processing is uploaded on a google drive, which can be accessed from the 
following link: https://drive.google.com/drive/folders/1SP02Nr2nP2UOgj--GKbbnunfV08AKpJS?usp=drive_link 

The Google drive contains two zip files. full_data.zip contains the full dataset. The other zip file, report_weights_and_test_set.zip
contains the weights and corresponding test set which were used when writing the report. Therefore these can be placed in the models and data folders respectively, 
then used in conjunction with the report_figures.ipynb notebook to recreate the exact figures in the report.

In order to run this code with the full dataset, download the zip file, extract the full_data.npz file and place it in the data folder.
Then change the DATA_MODE variable in the config.py file to 'full' and save. Then the unet_train.ipynb and model_eval.py
files can be used as before. For the full 100 epochs used in the project, this can take some time. On an Nvidia RTX3070 GPU, this took a little over an 
hour to train.
