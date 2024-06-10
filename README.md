# *UTILE-Redox* - Deep Learning based Tool for Autonomous 3D Bubble Analysis of Vanadium Flow Batteries from Synchrotron X-ray Imaging

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/workflow.jpg)


We present  an automated workflow using deep learning for the analysis of videos containing oxygen bubbles in PEM electrolyzers by 1. preparing an annotated dataset and training models in order to conduct semantic segmentation of bubbles and 2. automating the extraction of bubble properties for further distribution analysis.

The publication [UTILE-Redox - Deep Learning based Tool for Autonomous 3D Bubble Analysis of Vanadium Flow Batteries from Synchrotron X-ray Imaging]() will be available soon!


## Description
This project focuses on the deep learning-based automatic analysis of Vanadium Redox Flow Batteries (VRFB) Synchrotron X-ray tomographies. 
This repository contains the Python implementation of the UTILE-Redox software for automatic volume analysis, feature extraction, and visualization of the results.

The models we present in this work are trained on a specific use-case scenario of interest in VRFB bubble tomographies. Nevertheless, it is possible to fine-tune, re-train or employ another model suitable for your individual case if your data has a strong visual deviation from the presented data here, which was recorded and shown as follows:

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/capture.jpg)

## Model's benchmark
In our study, we trained several models to compare their prediction performance on unseen data. We trained specifically four different models on the same dataset composed by :

- U-Net 2D with a ResNeXt 101 backbone 
- Attention U-Net
- U-Net 3+
- Swin U-Net

We obtained the following performance results:

| Model                           | Precision [%] | Recall [%] | F1-Score [%] |
|---------------------------------|----------------|------------|--------------|
| U-Net with ResNeXt101 backbone  | 98             | 97         | 97           |
| Attention U-Net                 | 98             | 96         | 97           |
| U-Net 3+                        | 97             | 94         | 96           |
| Swin U-Net                      | 96             | 92         | 94           |

Since the F1-Scores are similar a visual inspection was carried out to find the best-performing model:

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/benchamark.jpg)


## Extracted features

### Membrane separation capabilites and 2D bubble density map from different planes

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/separationanddensity.jpg)

### Individual bubble shape analysis

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/individual.jpg)

### Bubbly membrane blockage

![](https://github.com/andyco98/UTILE-Redox/blob/main/images/blockade.jpg)

## Installation
In order to run the actual version of the code, the following steps need to be done:
- Clone the repository
- Create a new environment using Anaconda using Python 3.10
- Pip install the jupyter notebook library

    ```
    pip install notebook
    ```
- From your Anaconda console open jupyter notebook (just tip "jupyter notebook" and a window will pop up)
- Open the /UTILE-Redox/UTILE-Redox_prediction.ipynb file from the jupyter notebook directory
- Further instructions on how to use the tool are attached to the code with examples in the juypter notebook

## Dependencies
The following libraries are needed to run the program:

  ```
   pip install opencv-python, numpy, pillow, keras, tensorflow==2.11, matplotlib, scikit-image, pandas, tifffile, vtk, 

   ```
### Notes

Training and validation datasets and trained models are available at Zenodo: https://doi.org/10.5281/zenodo.11547023.