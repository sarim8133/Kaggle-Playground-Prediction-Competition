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

## Day 4: Decision Tree Tuning & LOOCV
I performed **hyperparameter tuning** on the **Decision Tree**, finding that a max_depth of 10 maximizes CV accuracy (98.46%). Depths beyond 10 caused overfitting. I also conducted a Leave-One-Out CV (LOOCV) study. Because LOOCV is computationally expensive $O(n^2)$, I ran it on a 500-sample subset. It scored 91.20%. The gap between the 5-Fold (98.46%) and LOOCV (91.2%) demonstrates how significantly model performance drops when deprived of large training data.

## Day 5 : Random Forest Ensemble

### Analysis: The Score Paradox
Despite a higher CV score (98.50%), the Random Forest (0.95945) underperformed the Decision Tree on the public leaderboard. This discrepancy suggests slight overfitting caused by the deeper **max_depth=15** setting in the Forest compared to the Tree's **max_depth=10**. It highlights the "Generalization Gap"—where a more complex model performs better on known data but fails to adapt to the specific noise within the public test set.

### Depth Sensitivity Study
I re-ran the Random Forest with **max_depth=10** to match the tuned Decision Tree. Surprisingly, the CV accuracy remained identical to the max_depth=15 run. This indicates that the ensemble naturally converges at a lower complexity, and additional depth provides no extra predictive power. This "saturation point" suggests that the most critical feature relationships are captured within the first 10 splits, making deeper trees redundant for this dataset.

## Day 6:

### Part 1: XGBoost(Gradient Boosting)

XGBoost produced a public score of **0.95945**, identical to the Random Forest. This suggests the models have reached a "feature ceiling" where performance is limited by the data itself rather than the algorithm. The fact that both ensembles underperform the simpler Decision Tree (0.96179) indicates that the test set favors a lower-complexity model. The ensembles are likely capturing "patterns" in the training data that are actually noise relative to the specific subset used for the public leaderboard.

### Part 2: LightGBM (Leaf-wise Boosting)

LightGBM achieved the highest public leaderboard score of **0.96628**. By utilizing leaf-wise tree growth, the model captured high-frequency patterns that level-wise models (XGBoost/Random Forest) missed. Interestingly, this model had a lower CV score **(98.38%)** than the Random Forest, providing a perfect example of the "Validation-Test Gap." This suggests that LightGBM’s specific regularization and gradient-based sampling allowed it to generalize better to the noise present in the Kaggle test set.

## Day 7 - Reaching Limits Through XGBoost FineTuning

### 1. The Architecture (15-Model Ensemble)
Instead of relying on one lucky model, the script built a fortress of 15 separate XGBoost models.
**5-Fold Cross-Validation** : It split your training data into 5 chunks, training on 4 and testing on 1, ensuring every single row of your data was evaluated without bias.
**3-Seed Averaging**: It ran that 5-Fold process three separate times using different random starting states (Seeds: 42, 2026, 777). This smoothed out any weird mathematical anomalies and stabilized the predictions.

### 🧬 2. The Feature Engineering
The script didn't just feed raw data into the trees; it transformed it:

**Digit Extraction**: It sliced your numerical features apart, extracting specific decimal digits (digit-4 to digit3) to expose underlying rounding patterns in the sensors.
**Ordered Target Encoding**: It translated your categorical data into probabilities (how likely a category is to result in Low, Medium, or High water need) while strictly preventing "data leakage" (preventing the model from cheating by looking at the validation answers).

### ⏱️ 3. The Hardware Stress Test
**Duration**: It ran for exactly 21,596 seconds (just under 6 hours).

**Memory Management**: It actively flushed the RAM (gc.collect()) after every single fold, preventing the Kaggle free-tier kernel from crashing under the weight of the massive datasets.

### 🏆 4. The Mathematical Victory (The Results)
**The Raw Score**: After averaging all 15 models, the pure, unweighted Out-Of-Fold (OOF) Balanced Accuracy was 0.978115.

**The Optuna Masterstroke:** The script then passed those raw predictions to Optuna. Optuna analyzed the mistakes and realized XGBoost was terribly under-predicting Class 3. It generated these exact multipliers:

**Class 1 (Low): 0.72x**

**Class 2 (Medium): 0.74x**

**Class 3 (High): 2.36x**

**The Final Score:** By aggressively boosting Class 3, your Balanced Accuracy skyrocketed to **0.980523**, and public score to **0.98083**.
