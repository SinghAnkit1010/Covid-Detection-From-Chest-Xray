**Chest X-ray COVID-19 Detection Web Application:**

This repository contains a web application that utilizes a convolutional neural network (CNN) model for detecting signs of COVID-19 in chest X-ray images. The model is built using transfer learning with VGG16 as the base model. The top three layers of VGG16 are removed and the remaining parameters are frozen. A fully connected layer is added on top of the VGG16 layer for classification.

**Dataset:**

The training dataset consists of three classes: COVID-19, Viral Pneumonia, and Normal. The dataset includes annotated chest X-ray images for each class, allowing the model to learn patterns and distinguish between different conditions.
  
**Model Training:**

During the model training process, we used the Adam optimizer, a popular optimization algorithm, to update the neural network weights. The model was trained for 20 epochs.To improve the model's generalization and robustness, we applied data augmentation techniques. Specifically, we randomly rotated the chest X-ray images within a certain range and randomly flipped them horizontally. These augmentations help introduce variety into the training data, making the model more capable of handling diverse image orientations and reducing overfitting..
The accuracy curve for training and validation is dataset is given below:
![download](https://github.com/SinghAnkit1010/Covid-Detection-From-Chest-Xray/assets/103994994/460d039a-0ce0-4964-b8a3-bf8a8370fe9f)


We get **0.92** recall on training data and **0.96** recall on validation data.

**Model Evaluation:**

To measure the performance of the model, we used accuracy and recall metrics. Recall was chosen as the primary metric of interest because it helps identify false negatives. Minimizing false negatives is crucial to prevent classifying positive cases (COVID-19 or viral pneumonia) as negative.
Furthermore model is evaluated on test data and we get following results:


|            | precision |  recall  | f1-score | support |
|------------|-----------|----------|----------|---------|
| COVID-19   |    1.00   |   1.00   |   1.00   |   26    |
| Viral Pneumonia |    0.87   |   0.65   |   0.74   |   20    |
| Normal     |    0.72   |   0.90   |   0.80   |   20    |
|            |           |          |          |         |
| accuracy   |           |          |   0.86   |   66    |
| macro avg  |    0.86   |   0.85   |   0.85   |   66    |
| weighted avg |  0.87   |   0.86   |   0.86   |   66    |


