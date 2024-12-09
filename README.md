## README: EuroSAT Image Classification Project

Project Overview
This project involves developing machine learning models to classify satellite images into ten land cover categories using the EuroSAT dataset. The primary goal is to leverage advanced models like Support Vector Machines (SVM) and Convolutional Neural Networks (CNN) to identify buildings and land cover types accurately. The project highlights its importance for applications in urban planning, disaster response, and environmental monitoring.

Objectives
Build a machine learning pipeline for satellite image classification.
Compare traditional models like SVM with deep learning models like CNN.
Evaluate the performance of models using metrics like accuracy, confusion matrix, and ROC-AUC.
Deploy the chosen CNN model for real-world usability with accessible APIs and interfaces.
Methodology
Dataset Overview:

EuroSAT Dataset: A public dataset containing 27,011 satellite images across ten categories.
Categories include: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, and SeaLake.
Images are resized to 64x64 pixels with 3 color channels (RGB), normalized to values between 0 and 1.
Data Preprocessing:

Resized and normalized images.
Labels encoded as integers (0-9).
Dataset split into 80% training and 20% testing, stratified by class.
Modeling:

Support Vector Machine (SVM):
Linear kernel with PCA for dimensionality reduction (100 components).
Accuracy: 45.48%.
Convolutional Neural Network (CNN):
Four convolutional layers with ReLU activations and Batch Normalization.
Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.1.
Sparse Categorical Cross Entropy for loss computation.
Achieved a test accuracy of 75.74% after 6 epochs.
Evaluation:

Metrics include Accuracy, Confusion Matrix, Classification Report, and Weighted AUC-ROC.
Visualization through heatmaps and ROC-AUC curves.
Results
SVM Results:
Accuracy: 45.48%.
Limited by computational inefficiency and high dimensionality of the dataset.
CNN Results:
Best model (4 layers, 6 epochs): Accuracy = 75.74%, Log Loss = 0.713.
Significant improvement over SVM due to automatic feature extraction and suitability for high-dimensional image data.
Challenges
SVM:
Computational inefficiency with 12,288 features per image.
High training time due to large dataset size.
CNN:
Hyperparameter tuning required extensive experimentation.
Training deep learning models was time-intensive, with each epoch taking ~10 minutes.
Potential Next Steps
Model Refinement:

Perform additional hyperparameter tuning (e.g., filter sizes, number of layers).
Experiment with data augmentation to improve generalization.
Validate the model on unseen datasets.
Deployment:

Save the trained CNN model in .h5 and .keras formats.
Develop a REST API using Flask or FastAPI for production deployment.
Create a web interface (e.g., Bubble.io) to allow users to upload images for classification.
Monitor model performance in production and retrain periodically to mitigate model drift.
Real-World Applications:

Collaborate with organizations like NASA, ESA, and JAXA for broader testing.
Extend the model for object detection or segmentation tasks.
Acknowledgements
Thanks to Break Through Tech Boston @MIT, Helen Bang (TA), and Challenge Advisors Hrishikesh Garud and Juliana Chyzhova for their guidance.
Tools used include Google Colab, TensorFlow/Keras, Python, and StackOverflow for problem-solving.
