Applied Text Mining AI/BA group 3 Documentation


This README describes the code that was produced for the course.

The folder 'workspace-negation-annotation' contains the annotations 
from assignment 1, where we had to annotate documents in the eHost tool.

The code for the rest of the assignments can be found in the folder 'source'.


Within source, all dataset files are located in the dataset folder. 

Now follows a general breakdown of the functionalities of the files. Most of these files have a corresponding
folder, where the save files corresponding to that python script are stored

Within each file, there is a more elaborate description of what the script does.

- data_preprocess.py -> takes a text file as input, processes it and adds the features as columns to the file.
- train.py -> reads the processed dataset file, initializes the CRF model and trains the model on the data.
- baseline_model.py and feature_selection_model.py -> inherits from the train.py model, but differes in the feature selection
- hyperopt.py -> makes use of the functions in the train.py class to perform the hyperopting of the model
- test.py -> given the data, it will load the model and make predictions on the test set. It then runs the evaluation to produce the scores.
- feature_coefficients_viz.py -> loads the model and extracts the state feature coefficients. It then produces a plot of the coefficients based on the features to shed insights into feature importance.
- data_exploration.py -> This file contains some functions to shed insights into some basic dataset properties, useful for data exploration.  

