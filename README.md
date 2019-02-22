# Sensorimotor Cross-Behavior Knowledge Transfer for Grounded Category Recognition

## Developer Environment
For our research, we used 64-bit Ubuntu 16.04 based computer with 16 GB RAM, Intel Core i7-7700 CPU (3.20 GHz x 8 cores) and NVIDIA GeForce GTX 1060 (3GB RAM, 1280 CUDA Cores).
The neural networks were implemented in widely used deep learning framework `TensorFlow 1.12` with GPU support (cuDNN 7, CUDA 9).

## Dependencies

`Python 3.5.6` is used for development and following packages are required to run the code:<br><br>
`pip install tensorflow-gpu==1.12.0`<br>
`pip install matplotlib==3.0.0`<br>
`pip install numpy==1.15.3`

## How to run the code?

Run `python main.py [mapping] [classifier]`

mapping: A2A, A2H, H2A, H2H <br>
classifier: KNN, SVM-RBF

Example: `python main.py H2H SVM-RBF`
