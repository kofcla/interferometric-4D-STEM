interferometric-4D-STEM
=======================
Code used for analysis of interference features in defocused CBED patterns recorded of twisted bilayer graphene as well as twist angle determination based on STEM-MAADF images.  Code for image simulation of these structures is also included. 

## step-by-step guide

1. Calculate the scale of ADF STEM images of monolayer graphene using the 
fourier-scale-calibrator by Jacob Madsen as shown in the notebook ``20230912_scaling_prep.ipynb``
and save it as dataframe as shown. This is important for the for the other functions.

2. Analyze your interferometric 4D STEM datasets using ``interferometric.py`` and 
``moire_class.py``. Examples of how to use this are given in the notebooks 
``analysis_interferometric_simulation.ipynb`` and ``analysis_interferometric_experiment.ipynb``. 
The two notebooks contain the analysis used for simulated and experimental data and differs mostly
in parameters for the different functions that depend on size of the images and so on.
Note that the savepaths and so on are just examples for my system and have to be changed to your 
datasets. 

## code files

### plot\_images\_with\_scale.py
Plots an image with scalebar and using a number of filters that can be specified and 
optionally saves it as pdf. 

### moire\_class.py
Analyzes a given image of some moiré pattern of tBLG to find the angle between the layers.
It works by finding the peaks in the FFT and calculating the angle between always the two 
closest spots and averaging over the (six) symmetric ones.
A number of parameters have to be changed to ensure the correct working for different images,
some examples for different fov are already implemented but have to be adjusted for most images.
The scale of the image has to be passed to the function for plotting reasons and the NionSwift
naming convention is asumed for finding the timestamp of the image.

### helper\_function.py
Contains functions that are used in the other classes like making scalebars for images and 
calculating angles. Pretty general.

### interferometric.py
Contains three classes to pre-process and subsequently analyze interferometric 4D-STEM datasets and
calculate physical properties like interlayer distance from cosine functions fitted to the 
interference fringes. Example usage shown in ``analysis_interferometric_simulation.ipynb`` and 
``analysis_interferometric_experiment.ipynb`` for simulated and real data.

## notebooks

### 20230912\_scaling\_prep.ipynb
Contains function definitions and useage to find the scale of STEM-MAADF images of graphene 
using fourier-scale-calibrator by Jacob Madsen and save it in a dataframe. Also some examples of 
analyzing moiré patterns using ``moire_class.py``. 

### analysis\_interferometric\_simulation.ipynb and analysis\_interferometric\_experiment.ipynb
Example usage of ``interferometric.py`` to analyze simulated and experimental interferometric 
4D-STEM datasets of tBLG.

### interferometric-simulation.ipynb
Uses *ab*TEM to simulate a MAADF image and interferometric 4D-STEM dataset. 

### coordinate\_analysis.ipynb
Shows how to calculate the interlayer distance and defocus variation from the coordinates of a 
structure file of tBLG.

### preparation\_of\_images.ipynb
Example usage of ``plot_images_with_scale.py`` to nicely plot and save images. 

## other files

### requirements.txt
The packages needed to make the code work. Can be installed using ``pip install -r requirements.txt``.
Please note that a virtual environment should be activated before installing the requirements to protect your system.

### README.md
This file.

### LICENSE
This repository is available under the ``GNU General Public License v3.0``, so you are free to use
change and distribute all contents. The exact license text can be found in the ``LICENSE`` file.
