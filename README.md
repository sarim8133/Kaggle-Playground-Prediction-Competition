# Kaggle-Playground-Prediction-Competition
Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.  Your Goal: Predict the irrigation need.

## Day 1 : Data Exploration & Baseline Submission

Sign up to Kaggle, join S6E4 "Predicting Irrigation Need", accept rules
Download train.csv, test.csv, sample_submission.csv from the Data tab
Run the pipeline: class distribution, missing values, correlation heatmap
Understand columns — soil moisture, temp, humidity, etc. Read the dataset description
Submit baseline
— all majority class 
**Goal**: know the data shape and see your first Kaggle score before anything else

**Low**: 369,917 (Majority) , **Medium**: 239,074 , **High**: 21,009

Given the severe class imbalance, I implemented a "Zero-R" (majority-class) baseline rather than training an algorithm right out of the gate. Since "Low" is by far the most frequent category in the training data, my baseline script blindly predicts "Low" for every single instance in the test dataset hence my Keggle Score Baseline was set as **0.333**.

## Day 2 : Baseline models — Decision Tree & Naive Bayes

### Decision Tree Results

I trained a Decision Tree model with an initial 5-Fold CV accuracy of **98.01%**. It performs excellently on majority classes but has lower precision (82%) on the minority "High" class. Tuning the **max_depth parameter** showed that a depth of 10 maximizes CV accuracy at **98.46%**. I also ran **LOOCV** on a 500-sample subset, scoring **91.2%**. Finally, I retrained this tuned model on 100% of the training data to generate today's submission.

### Naive Bayes Results
I trained a Gaussian Naive Bayes model on the scaled features, achieving a **5-Fold CV accuracy** of **82.67%**. While it performed decently on the majority "Low" class (90% recall), it struggled significantly with the minority "High" class, dropping to just **42% recall**. Compared to the Decision Tree's 98.46%, GNB underperforms on this dataset. I generated a final submission to log its public leaderboard score for comparison.

### Wrapping up Day 2!
Decision Tree gave a public score of **0.96179** and Naive Bayes gave only a **0.68609**.

## Day 3 : Logistic Regression & K-Means classifier

### Logistic Regression
Logistic Regression achieved a CV accuracy of **74.88%** and a Kaggle score of **0.61265**. The model showed significant underfitting compared to tree-based methods. Tuning the regularization parameter `C` had zero impact on accuracy, confirming that the relationship between soil/weather features and irrigation need is fundamentally **non-linear**. The linear decision boundaries were too rigid to capture the complex patterns required for high-accuracy predictions.

### K-Means
K-Means produced a validation accuracy of **58.72%**, the lowest of all models tested. Because K-Means is unsupervised, it grouped data by spatial distance rather than labels. Due to the high class imbalance, the majority class ("Low") dominated every cluster, leading the mapping strategy to predict "Low" for all instances. This **"failed attempt"** proves that distance-based clustering without label guidance is ineffective for this specific classification task.

