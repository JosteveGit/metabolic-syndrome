# Deep Learning for Detection, Segmentation, and Quantification of Fat in CT Scans

This repository is a comprehensive collection of scripts, Jupyter notebooks, and assets developed during my dissertation project, which focuses on applying deep learning techniques to advance metabolic health analysis by detecting, segmenting, and quantifying visceral and superficial fat in CT scans.

## Project Structure

Below is an overview of the directory structure and contents of this repository:

- `1.dataset`: 
  - `dataset.md`: A detailed explanation of the dataset used, including privacy considerations in accordance with NHS data handling policies.

- `2.pre-processing`: 
  - A series of scripts for converting medical imaging data from DICOM and NRRD formats to image matrices, and for preparing and cleansing the data for subsequent processing.

- `3.augmentation`: 
  - Python scripts designed to augment the dataset through various transformations, ensuring a robust model training process.

- `4.training`: 
  - Contains Jupyter notebooks for each phase of neural network model training, including different architectures and approaches explored in this project.

- `5.fat-quantification`: 
  - A script dedicated to the quantification aspect of the project, turning segmented data into actionable fat quantifications.

- `6.web-inference-platform`: 
  - Assets for a web-based inference platform, intended to demonstrate the practical application of the trained models.

