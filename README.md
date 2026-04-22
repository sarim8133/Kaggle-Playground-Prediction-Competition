# Kaggle-Playground-Prediction-Competition
Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.  Your Goal: Predict the irrigation need.

## Day 1

Sign up to Kaggle, join S6E4 "Predicting Irrigation Need", accept rules
Download train.csv, test.csv, sample_submission.csv from the Data tab
Run the pipeline: class distribution, missing values, correlation heatmap
Understand columns — soil moisture, temp, humidity, etc. Read the dataset description
Submit baseline
— all majority class 
**Goal**: know the data shape and see your first Kaggle score before anything else

**Data Exploration & Baseline Submission**

**Low**: 369,917 (Majority)** , **Medium**: 239,074 , **High**: 21,009

Given the severe class imbalance, I implemented a "Zero-R" (majority-class) baseline rather than training an algorithm right out of the gate. Since "Low" is by far the most frequent category in the training data, my baseline script blindly predicts "Low" for every single instance in the test dataset.
