import os
import random
import warnings
import gc  # <-- The RAM Janitor
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')
print("Initiating the Crash-Proof Grandmaster Pipeline: 5-Fold CV + 3-Seed Averaging + OrderedTE + Optuna...")

# --- 1. RANDOM SEED ---
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(seed=2026)

# --- 2. DATA LOADING & AUTO-HUNTER ---
target_col = 'Irrigation_Need'
num_classes = 3
drop_cols = ['id']

# Load base data
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/train.csv").drop(drop_cols, axis=1)
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/test.csv").drop(drop_cols, axis=1)
sub = pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/sample_submission.csv")

print("Hunting for the Original Clean Dataset...")
original_df = pd.DataFrame()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv') and filename not in ['train.csv', 'test.csv', 'sample_submission.csv']:
            file_path = os.path.join(dirname, filename)
            try:
                tmp = pd.read_csv(file_path)
                if target_col in tmp.columns:
                    print(f"SUCCESS! Found clean data: {file_path}")
                    original_df = pd.concat([original_df, tmp], ignore_index=True)
            except:
                pass

if not original_df.empty:
    if 'id' in original_df.columns:
        original_df = original_df.drop(columns=['id'])
    train = pd.concat([train, original_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"Dataset successfully expanded! Total training rows: {len(train)}")

# Target Mapping
target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
idx2target = {v: k for k, v in target2idx.items()}
train[target_col] = train[target_col].map(target2idx)

CATS = [c for c in test.columns if train[c].dtype == object]
NUMS = [c for c in test.columns if c not in CATS]

# --- 3. FEATURE ENGINEERING (Digit Hacker Trick) ---
print("Applying Feature Engineering (Digit Extraction)...")
M = train[NUMS].max()

def FE(df): 
    for c in NUMS:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    return df 

train = FE(train)
test = FE(test)

DROP = [c for c in test.columns if test[c].nunique() == 1]
train.drop(DROP, axis=1, inplace=True)
test.drop(DROP, axis=1, inplace=True)

CATEGORY = CATS + [c for c in test.columns if 'digit' in c]
for c in CATEGORY:
    freq = train[c].value_counts()
    mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
    mapping_default = len(mapping)
    train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
    test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))

FEATURES = CATEGORY + NUMS

# --- 4. SAMPLE WEIGHTS ---
unique, counts = np.unique(train[target_col].values, return_counts=True)
count_dict = dict(zip(unique, counts))
avg_count = len(train) / len(unique)
weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
sample_weights = np.array([weights_dict[y] for y in train[target_col]])

# --- 5. ORDERED TARGET ENCODING CLASS ---
class OrderedTE():
    def __init__(self, a=1):
        self.a = a
        
    def fit(self, train, category_cols=[], target_col='target'):
        self.train = train
        self.target_col = target_col
        self.category_cols = category_cols
        self.classes_ = sorted(train[target_col].unique())
        self.num_classes_ = len(self.classes_)
        self.global_prior_ = train[target_col].value_counts(normalize=True).sort_index().values
        
        for c in self.category_cols:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y_binary = (train[target_col] == cls).astype(int)
                df = train[[c]].copy()
                df['y'] = y_binary.values
                df['cnt'] = 1
                df['cum_cnt'] = df.groupby(c)['cnt'].cumsum() - df['cnt']
                df['cum_sum'] = df.groupby(c)['y'].cumsum() - df['y']
                smooth_prior = self.a * self.global_prior_[k]
                te_col = f'{c}_TE_cls{cls}'
                df[te_col] = (df['cum_sum'] + smooth_prior) / (df['cum_cnt'] + self.a)
                df.loc[df['cum_cnt'] == -1, te_col] = self.global_prior_[k]
                self.train[te_col] = df[te_col].values
                
                stats_df = df.groupby(c)['y'].agg(['count', 'sum']).reset_index()
                stats_df.columns = [c, f'{c}_count_cls{cls}', f'{c}_sum_cls{cls}']
                stats_df[f'{c}_prior_cls{cls}'] = self.global_prior_[k]
                stats_list.append(stats_df)
            
            combined_stats = stats_list[0]
            for i in range(1, len(stats_list)):
                combined_stats = combined_stats.merge(stats_list[i], on=c, how='outer')
            setattr(self, f'{c}_stats', combined_stats)
        return self.train
    
    def transform(self, test):
        for c in self.category_cols:
            stats_df = getattr(self, f'{c}_stats')
            test = test.merge(stats_df, on=c, how='left')
            for k, cls in enumerate(self.classes_):
                te_col = f'{c}_TE_cls{cls}'
                count_col = f'{c}_count_cls{cls}'
                sum_col = f'{c}_sum_cls{cls}'
                prior_col = f'{c}_prior_cls{cls}'
                if count_col in test.columns:
                    test[te_col] = (test[sum_col] + self.a * test[prior_col]) / (test[count_col] + self.a)
                    test[te_col] = test[te_col].fillna(test[prior_col])
                    test.drop([count_col, sum_col, prior_col], axis=1, inplace=True)
                else:
                    test[te_col] = self.global_prior_[k]
        return test

# --- 6. MEMORY OPTIMIZATION ---
def reduce_mem_usage(df:pd.DataFrame, float16_as32:bool=True)->pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)!='category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32: df[col] = df[col].astype(np.float32)
                    else: df[col] = df[col].astype(np.float16)  
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    return df

# --- 7. THE 3-SEED, 5-FOLD ENSEMBLE ---
print("\nInitializing Grandmaster XGBoost Ensemble (3 Seeds x 5 Folds)...")
X = train.drop([target_col], axis=1)
y = train[target_col]
test_X = test.copy()

# Master prediction arrays
final_oof_preds = np.zeros((len(y), num_classes))
final_test_preds = np.zeros((len(test_X), num_classes))

SEEDS = [42, 2026, 777] 
n_folds = 5             # <-- Lowered to 5 to prevent OOM timeouts

xgb_params = {
    'max_depth': 4,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 1024,
    'learning_rate': 0.1,
    'n_jobs': -1,
    'enable_categorical': True,
    'alpha': 5,
    'reg_lambda': 5,
    'max_leaves': 30,
    'min_child_weight': 2,
    'tree_method': 'hist',
    'max_bin': 10000,
    'device': 'cuda'
}

for seed in SEEDS:
    print(f"\n====================================")
    print(f"  STARTING SEED: {seed}")
    print(f"====================================")
    
    xgb_params['random_state'] = seed
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    seed_oof_preds = np.zeros((len(y), num_classes))
    seed_test_preds = np.zeros((len(test_X), num_classes))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  --> Training Fold {fold+1}/{n_folds} (Seed {seed})")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_weights = sample_weights[train_idx]

        te = OrderedTE()
        full_df = pd.concat((X_train, y_train), axis=1)
        full_df['weight'] = train_weights
        
        # <-- Fixed data multiplier: range(4) down to range(2) -->
        te_train = pd.concat([te.fit(full_df.sample(frac=1, random_state=seed+i), 
                                     category_cols=FEATURES, target_col=target_col) for i in range(2)])
        
        X_train = te_train.drop([target_col, 'weight'], axis=1)
        y_train = te_train[target_col]
        train_weights = te_train['weight']
        
        X_val = te.transform(X_val)
        X_test_fold = te.transform(test_X)

        X_train.drop(CATS, axis=1, inplace=True)
        X_val.drop(CATS, axis=1, inplace=True)
        X_test_fold.drop(CATS, axis=1, inplace=True)

        X_train = reduce_mem_usage(X_train)
        X_val = reduce_mem_usage(X_val)
        X_test_fold = reduce_mem_usage(X_test_fold)

        model = XGBClassifier(**xgb_params)
        # verbose=200 so you can actually watch the GPU rip through the trees
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=train_weights, verbose=200) 
        
        seed_oof_preds[val_idx] = model.predict_proba(X_val)
        seed_test_preds += model.predict_proba(X_test_fold) / n_folds

        # <-- GARBAGE COLLECTION: Flushes the RAM completely after each fold -->
        del X_train, y_train, X_val, te_train, X_test_fold, full_df, model
        gc.collect()
        # -------------------------------------------------------------------
        
    # Add this seed's predictions to the master arrays
    final_oof_preds += seed_oof_preds / len(SEEDS)
    final_test_preds += seed_test_preds / len(SEEDS)
    
    # Print the score for the current seed
    seed_acc = balanced_accuracy_score(y, np.argmax(seed_oof_preds, axis=1))
    print(f"  [Seed {seed} Complete] OOF Balanced Accuracy: {seed_acc:.6f}")

print(f"\n✅ GRANDMASTER ENSEMBLE COMPLETE")
print(f"Total Combined OOF Balanced Accuracy: {balanced_accuracy_score(y, np.argmax(final_oof_preds, axis=1)):.6f}")

# --- 8. OPTUNA CLASS WEIGHT OPTIMIZATION ---
print("\nRunning Optuna Class Weight Optimizer on Ensembled Probabilities...")
def objective(trial):
    cw1 = trial.suggest_float('cw1', 0.5, 3.0)
    cw2 = trial.suggest_float('cw2', 0.5, 3.0)
    cw3 = trial.suggest_float('cw3', 0.5, 3.0)
    
    class_weights = np.array([cw1, cw2, cw3])
    adjusted_probs = final_oof_preds * class_weights
    adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
    
    acc = balanced_accuracy_score(y, np.argmax(adjusted_probs, axis=1))
    return acc

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
optuna.logging.set_verbosity(optuna.logging.WARNING) 
study.optimize(objective, n_trials=300, show_progress_bar=False)

print(f"\nFinal Post-Optuna Balanced Accuracy: {study.best_value:.6f}")
print("Optimal Weights:", study.best_params)

# --- 9. FINAL PREDICTIONS ---
best_cw = np.array([study.best_params['cw1'], study.best_params['cw2'], study.best_params['cw3']])
final_test_probs = final_test_preds * best_cw
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

final_test_preds_classes = np.argmax(final_test_probs, axis=1)
sub[target_col] = final_test_preds_classes
sub[target_col] = sub[target_col].map(idx2target)

sub.to_csv("submission_grandmaster_ensemble.csv", index=False)
print("\n🎉 SUCCESS! Upload 'submission_grandmaster_ensemble.csv' and check the leaderboard!")

# --- 10. AUTO-SHUTDOWN TO SAVE GPU QUOTA ---
import os
import time

print("\nScript finished! Submission saved.")
print("Auto-shutting down the Kaggle server in 10 seconds to save GPU hours...")

# Give the filesystem a few seconds to safely finish writing the CSV to disk
time.sleep(10)

# The Linux "kill all processes" command. This instantly terminates your Kaggle session.
os.system("kill -9 -1")