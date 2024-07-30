# Predicting-Employee-Attrition
# Overview
This project involves conducting an analysis on the HR Analytics dataset to understand the factors contributing to employee attrition. The analysis includes data preprocessing, model training, evaluation, and visualization.

# Dataset
The dataset used in this project is HR_Analytics.csv, which contains various features related to employees, including their demographics, job roles, satisfaction levels, and whether they have left the company (attrition).

# Project Structure
1.Data Loading and Inspection

2.Load the dataset using Pandas.
Display basic information, statistics, and initial rows of the dataset.
Data Preprocessing

3.Handle missing values by dropping rows with missing data.
Convert categorical variables into dummy/indicator variables.
Scale features using StandardScaler from Scikit-learn.
Data Splitting

4.Split the dataset into training and testing sets using train_test_split.
Model Training

5.Train a Random Forest Classifier on the training data.
Model Evaluation

Evaluate the model using confusion matrix, classification report, and accuracy score.
Visualize the results using a confusion matrix heatmap and a pairplot for selected features.
# Libraries Used
pandas, matplotlib.pyplot, seaborn, sklearn.model_selection, sklearn.preprocessing, sklearn.ensemble, sklearn.metrics
