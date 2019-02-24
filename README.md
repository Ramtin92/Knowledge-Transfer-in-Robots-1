# Sensorimotor Cross-Behavior Knowledge Transfer for Grounded Category Recognition

## Development Environment
For our research, we used 64-bit Ubuntu 16.04 based computer with 16 GB RAM, Intel Core i7-7700 CPU (3.20 GHz x 8 cores) and NVIDIA GeForce GTX 1060 (3GB RAM, 1280 CUDA Cores).
The neural networks were implemented in widely used deep learning framework `TensorFlow 1.12` with GPU support (cuDNN 7, CUDA 9).

## Dependencies

`Python 3.5.6` is used for development and following packages are required to run the code:<br><br>
`pip install tensorflow-gpu==1.12.0`<br>
`pip install matplotlib==3.0.0`<br>
`pip install numpy==1.15.3`

## How to run the code?

Run: `python main.py [mapping] [classifier]`

mapping: A2A, A2H, H2A, H2H <br>
classifier: KNN, SVM-RBF

Example: `python main.py H2H SVM-RBF`

## Experiment Pipeline 

- The source robot interacts with all the 20 categories (highlighted in solid red line), but the target robot interacts with only 15 categories (highlighted in solid blue line).

<img src="pics/Slide1.PNG" alt="drawing" width="600px"/>

- The objects of 15 categories shared by both the robots are used to train the encoder-decoder network that learns to projects the sensory signal of the source robot to the target robot.

<img src="pics/Slide2.PNG" alt="drawing" width="600px"/>

- Subsequently, the trained encoder-decoder network is used to generate “fake” sensory signals for the other 5 object categories (highlighted in dashed blue line) that the target robot did not interact with by projecting the sensory signal of the source robot.

<img src="pics/Slide3.PNG" alt="drawing" width="600px"/>

- Once the features are projected, an category recognition classifier is trained using the projected
data from the source context (i.e., how well it would do if it transfered knowledge from the source robot).

<img src="pics/Slide4.PNG" alt="drawing" width="600px"/>

- Additional category recognition classifier is trained using the ground truth data produced by the target robot (i.e., the best the target robot could do if it had explored all the objects) for comparison.

<img src="pics/Slide5.PNG" alt="drawing" width="600px"/>

## Results

## k-Nearest Neighbors (3-NN)

### Accuracy

<img src="Results/KNN/Top_5_Minimum_Accuracy_Delta_Mappings_(KNN).png" alt="drawing" width="600px"/>
<img src="Results/KNN/Top_5_Maximum_Accuracy_Delta_Mappings_(KNN).png" alt="drawing" width="600px"/>

### RMSE loss vs Accuracy

<img src="Results/KNN/RMSELossvsAccuracyKNN_Train_3.png" alt="drawing" width="600px"/>

### Accuracy Delta

<img src="Results/KNN/Classification_Loss_A2A_2.png" alt="drawing" width="600px"/>
<img src="Results/KNN/Classification_Loss_A2H_2.png" alt="drawing" width="600px"/>
<img src="Results/KNN/Classification_Loss_H2A_2.png" alt="drawing" width="600px"/>
<img src="Results/KNN/Classification_Loss_H2H_2.png" alt="drawing" width="600px"/>

### RMSE loss vs Accuracy Delta

<img src="Results/KNN/RMSELossvsAccuracyLossKNN_Train_3.png" alt="drawing" width="600px"/>

## Support Vector Machine (SVM-RBF) 

### Accuracy

<img src="Results/SVM/Top_5_Minimum_Accuracy_Delta_Mappings_(SVM).png" alt="drawing" width="600px"/>
<img src="Results/SVM/Top_5_Maximum_Accuracy_Delta_Mappings_(SVM).png" alt="drawing" width="600px"/>

### RMSE loss vs Accuracy

<img src="Results/SVM/RMSELossvsAccuracySVM-RBF_Train_3.png" alt="drawing" width="600px"/>

### Accuracy Delta

<img src="Results/SVM/Classification_Loss_A2A_2.png" alt="drawing" width="600px"/>
<img src="Results/SVM/Classification_Loss_A2H_2.png" alt="drawing" width="600px"/>
<img src="Results/SVM/Classification_Loss_H2A_2.png" alt="drawing" width="600px"/>
<img src="Results/SVM/Classification_Loss_H2H_2.png" alt="drawing" width="600px"/>

### RMSE loss vs Accuracy Delta

<img src="Results/SVM/RMSELossvsAccuracyLossSVM-RBF_Train_3.png" alt="drawing" width="600px"/>


