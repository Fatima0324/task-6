# K-Nearest Neighbors (KNN) Classification

This repository contains the implementation of a K-Nearest Neighbors (KNN) classifier for the AI & ML Internship Task 6.

## Overview

This project demonstrates the application of the KNN algorithm to a classification problem using the Iris dataset. The implementation focuses on understanding how KNN works, the importance of parameter selection (K value), feature normalization, and model evaluation techniques.

## Features

- Data loading and preprocessing
- Feature normalization using StandardScaler
- Implementation of KNN classifier using scikit-learn
- Systematic evaluation of different K values
- Model evaluation using accuracy, confusion matrix, and classification report
- Visualization of decision boundaries for different feature pairs
- Comprehensive documentation and explanations

## Repository Structure

- `knn_classification.py`: Main Python script containing the complete implementation
- `knn_k_values_performance.png`: Graph showing the relationship between K values and model performance
- `knn_confusion_matrix.png`: Visualization of the confusion matrix for the best model
- `knn_decision_boundaries.png`: Visualization of decision boundaries for different feature pairs
- `README.md`: Project documentation (this file)

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to Run

1. Clone this repository
2. Install the required dependencies: `pip install numpy pandas matplotlib scikit-learn`
3. Run the script: `python knn_classification.py`

## Results

The script performs the following steps:
1. Loads the Iris dataset
2. Normalizes the features using StandardScaler
3. Trains KNN models with different K values (1 to 30)
4. Evaluates models using training, test, and cross-validation accuracy
5. Identifies the optimal K value
6. Performs detailed evaluation of the best model
7. Visualizes decision boundaries for different feature pairs

## Key Findings

- The optimal K value for the Iris dataset is determined through cross-validation
- Feature normalization significantly improves model performance
- KNN achieves high accuracy on the Iris dataset, demonstrating its effectiveness for this type of classification task
- Decision boundaries visualizations show how the model classifies the data in different feature spaces

## Answers to Interview Questions

### 1. How does the KNN algorithm work?

KNN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority class among their k-nearest neighbors. The algorithm works as follows:
- Store all training examples in memory
- For a new input, calculate the distance between it and all training examples
- Select the K training examples that are nearest to the input
- Assign the input to the class that appears most frequently among these K neighbors
- For regression tasks, the output would be the average of the K neighbors' values

### 2. How do you choose the right K?

The optimal K value can be selected through:
- Cross-validation: Testing different K values and selecting the one that gives the best validation performance
- Square root rule: K ≈ √n where n is the number of training samples
- Odd vs. Even: Using odd K values to avoid ties in binary classification
- Domain knowledge: Considering the specific requirements of the problem

Too small K values lead to noisy decision boundaries (overfitting), while too large K values result in over-smoothed boundaries (underfitting).

### 3. Why is normalization important in KNN?

Normalization is crucial in KNN because:
- KNN uses distance metrics to find nearest neighbors
- Features with larger scales would dominate the distance calculation
- Normalized features ensure all variables contribute equally to distance calculations
- Without normalization, features with larger numeric ranges would have disproportionate influence on the model
- Normalization improves both model accuracy and convergence speed

### 4. What is the time complexity of KNN?

KNN has the following time complexity:
- Training complexity: O(1) - KNN is a lazy learning algorithm with no training phase (just stores the data)
- Testing complexity: O(n·d) for each prediction, where n is the number of training examples and d is the number of features
- Space complexity: O(n·d) to store the entire training dataset
- For large datasets, optimization techniques like KD-trees can reduce prediction time to O(log n)

### 5. What are pros and cons of KNN?

Pros:
- Simple to understand and implement
- No training phase (lazy learning)
- Naturally handles multi-class problems
- No assumptions about data distribution (non-parametric)
- Can be effective for complex decision boundaries

Cons:
- Computationally expensive for large datasets
- Requires feature scaling/normalization
- Sensitive to irrelevant features and the curse of dimensionality
- Storage requirements for the entire training dataset
- Prediction speed can be slow for large datasets
- Optimal K value selection can be challenging

### 6. Is KNN sensitive to noise?

Yes, KNN is sensitive to noise for several reasons:
- Noisy samples can significantly influence predictions for nearby points
- With small K values, noisy data points have higher impact
- As K increases, sensitivity to noise decreases, but at the cost of potentially missing important patterns
- Preprocessing techniques like outlier removal and smoothing can help mitigate noise issues
- Distance weighting (giving higher weights to closer neighbors) can reduce the impact of noisy samples

### 7. How does KNN handle multi-class problems?

KNN naturally handles multi-class classification:
- It simply counts the number of neighbors belonging to each class
- The predicted class is the one with the most neighbors among the K nearest
- No modification to the algorithm is needed to support multiple classes
- In case of ties, they can be broken by:
  - Reducing K by 1
  - Using weighted voting based on distances
  - Selecting the class with the minimum average distance

### 8. What's the role of distance metrics in KNN?

Distance metrics determine how "similarity" is calculated in KNN:
- Euclidean distance: Most common, works well for continuous variables
- Manhattan distance: Better for discrete or binary features
- Minkowski distance: Generalization of Euclidean and Manhattan distances
- Hamming distance: Suitable for categorical variables
- Cosine similarity: Useful when the magnitude of vectors is less important than their orientation

The choice of distance metric should match the characteristics of the data and the problem domain. Different metrics emphasize different aspects of similarity between data points.
