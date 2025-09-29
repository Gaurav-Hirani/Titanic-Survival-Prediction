🚢 Titanic Survival Prediction: A Comparative ML Study

Project Overview

This project applies fundamental Machine Learning techniques to the classic Titanic survival dataset to predict which passengers survived the disaster. The goal is to compare the performance of multiple classification algorithms after performing essential data cleaning and preprocessing steps.

The entire workflow, from data loading and cleaning to model training and evaluation, is contained within the Jupyter Notebook, titanic.ipynb.

🛠️ Methodology and Workflow
1. Data Loading and Initial Cleaning
Data Source: The dataset is loaded directly from the seaborn library.
Feature Selection: Several columns were dropped to simplify the model and handle missing data efficiently: deck (high NaN count), and redundant features like alive, class, who, adult_male, and embark_town.
Missing Value Handling:
Missing age values were imputed using the column's mean.
Rows with missing embarked values were dropped (only 2 records).

2. Feature Engineering & Scaling
Categorical Encoding: The sex and embarked features were converted into numerical format using Label Encoding.
Data Preparation: The data was split into training (80%) and testing (20%) sets with a fixed random_state=42 for reproducibility.
Feature Scaling: StandardScaler was applied to the training and test sets. This step is crucial for distance-based models (like KNN and SVM) to prevent features with large ranges (e.g., fare) from dominating the distance calculations.

3. Model Training and Evaluation
Four distinct classification models from scikit-learn were trained and evaluated on the processed data.

Performance Summary (Test Accuracy)

The models were compared based on their performance on the unseen test data:

K-Nearest Neighbors (KNN): 82.02% Accuracy (Scaling Applied: Yes)

Support Vector Machine (SVM): 81.46% Accuracy (Scaling Applied: Yes)

Logistic Regression: 80.34% Accuracy (Scaling Applied: No)

Decision Tree Classifier: 80.34% Accuracy (Scaling Applied: Yes)

Gaussian Naive Bayes: 77.53% Accuracy (Scaling Applied: No)

Key Insights

The KNN model achieved the highest overall accuracy at 82.02%.

The SVM model demonstrated the highest Recall for predicting actual survivors (Class 1), achieving a score of 0.78.

The models that utilized Feature Scaling (KNN, SVM, Decision Tree) generally outperformed the unscaled models.

🚀 Getting Started
Prerequisites

To run this project, you need a Python environment with the following libraries installed:

pip install numpy pandas seaborn matplotlib scikit-learn jupyter

Running the Notebook

Clone this repository:

git clone [https://github.com/YourUsername/Titanic-Survival-Prediction.git](https://github.com/Gaurav-Hirani/Titanic-Survival-Prediction.git)
cd Titanic-Survival-Prediction

Launch Jupyter Lab or Jupyter Notebook:

jupyter notebook

Open and run the cells in the titanic.ipynb file.

Future Enhancements
Hyperparameter Tuning: Use GridSearchCV to optimize parameters (e.g., the optimal n_neighbors for KNN or C for SVM).

Advanced Encoding: Implement One-Hot Encoding for the embarked feature to avoid introducing misleading ordinal relationships.

Feature Engineering: Create composite features like Family_Size (sibsp + parch + 1) for potentially higher predictive power.

