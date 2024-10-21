Tamil Handwritten Character Recognition Using CNN-SVM Hybrid Model
Overview
This project implements a hybrid deep learning model combining Convolutional Neural Networks (CNNs) for feature extraction and Support Vector Machines (SVMs) for classification. The goal is to recognize handwritten Tamil vowels from grayscale images.

Dataset
The dataset consists of handwritten Tamil vowels, with 11 categories representing different characters. The images are grayscale and have been preprocessed using various techniques, including:

Rescaling (Normalization to [0, 1])
Data Augmentation (Rotation, translation, and scaling)
Image Denoising (Optional, applied as necessary)
The dataset has been split into training and validation sets, with around 250+ samples per class in the training set. The validation set has a slightly imbalanced distribution across categories.

Model Architecture
The hybrid model consists of the following stages:

1. Preprocessing:
Images are rescaled and augmented to ensure robustness against real-world variations in handwriting styles.
2. Convolutional Neural Network (CNN):
CNN layers are used to extract features from the input images.
Includes convolutional layers, batch normalization, max pooling, and dropout layers.
Extracted features are passed to the next stage for classification.
3. Support Vector Machine (SVM):
The output features from the CNN are passed to an SVM classifier.
SVM, with a linear kernel, performs the final classification, predicting which Tamil character the image represents.
Results
Accuracy: 97.17%
Precision: 0.97
Recall: 0.97
F1 Score: 0.97
These results indicate that the hybrid CNN-SVM model is effective in classifying handwritten Tamil characters with high accuracy and efficiency.


Dataset Inference
The dataset contains 11 categories with nearly equal representation in the training set.
Data augmentation ensures robust learning, improving the model's generalization to unseen handwriting styles.
The validation dataset shows slight class imbalance, but this is not significant enough to affect the overall model performance.
Future Improvements
Attention mechanisms could be introduced to improve the model’s focus on critical features of similar characters.
Further data collection: Expanding the dataset with more handwritten samples can help improve generalization even further.
License
This project is licensed under the MIT License.
