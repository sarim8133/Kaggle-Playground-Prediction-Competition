"""
====================================================================
Kaggle Playground Series S6E4 — Predicting Irrigation Need
Multiclass Classification Pipeline
====================================================================
Assignment 3 — Machine Learning
Covers: Decision Tree, Naive Bayes, K-Means (as classifier),
        Logistic Regression, Random Forest, XGBoost/LGBM,
        K-Fold CV, LOOCV, Confusion Matrices, Final Submission
====================================================================
SETUP:
  pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
  Place train.csv and test.csv in the same folder, then run:
  python irrigation_ml_pipeline.py
====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# ===== SET THE MODEL YOU WANT TO RUN + SUBMIT =====
# Day 1: "baseline"  →  majority-class submission (no model training)
# Day 2: "dt"        →  Decision Tree  (run → submit)
#        "gnb"       →  Naive Bayes    (run → submit)
# Day 3: "lr"        →  Logistic Regression (+ KMeans eval alongside)
# Day 5: "rf"        →  Random Forest
# Day 6: "xgb"       →  XGBoost
#        "lgbm"      →  LightGBM
# Day 8: "best"      →  Runs ALL models, auto-picks highest accuracy
# Day 9: "optuna_lgbm" → Optuna-optimized LightGBM
# Day 10: "champion" -> The 0.9845 Champion (Manual Best)
# Day 11: "hybrid"   -> Hybrid Feature Engineering (The 1.000 Target)
# Day 12: "de_lgbm"  -> Differential Evolution Optimized LightGBM

MODEL_TO_SUBMIT = "de_lgbm"

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Skipping.")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARN] LightGBM not installed. Skipping.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] Optuna not installed. Run 'pip install optuna' for hyperparameter tuning.")

# ── Config ───────────────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5
RESULTS = {}
os.makedirs("confusion_matrices", exist_ok=True)
np.random.seed(SEED)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Loading Data (WITH ORIGINAL DATA)")
print("="*60)

# Load the competition data
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# Load the newly found original data
try:
    orig_data = pd.read_csv("irrigation_prediction.csv")
    
    # Drop the 'id' column from the original data if it exists, 
    # so it matches the competition train data structure
    if 'id' in orig_data.columns:
        orig_data = orig_data.drop(columns=['id'])
    
    # Stack the original data on top of the synthetic training data
    train = pd.concat([train, orig_data], ignore_index=True)
    
    # Remove any exact duplicate rows that might confuse the model
    train = train.drop_duplicates(ignore_index=True)
    print("Successfully merged original data!")
except FileNotFoundError:
    print("[WARN] Original data 'irrigation_prediction.csv' not found. Proceeding with competition data only.")

print(f"NEW Train shape (with Original) : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"Columns     : {list(train.columns)}")

# Auto-detect target (last column or 'target' / 'irrigation_need')
TARGET = None
for candidate in ["target", "irrigation_need", "Irrigation_Need", "label", "Label"]:
    if candidate in train.columns:
        TARGET = candidate
        break
if TARGET is None:
    TARGET = train.columns[-1]
print(f"Target column → '{TARGET}'")
print(f"Target dist   :\n{train[TARGET].value_counts()}")

# ── 2. EDA ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — EDA")
print("="*60)

print(train.describe())
missing_train = train.isnull().sum()
print(f"\nMissing values:\n{missing_train[missing_train > 0]}")

plt.figure(figsize=(7,4))
train[TARGET].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Target Class Distribution")
plt.xlabel("Class"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("confusion_matrices/00_class_distribution.png", dpi=120)
plt.close()
print("[saved] 00_class_distribution.png")

# Correlation heatmap (numeric only)
num_cols_eda = train.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols_eda) > 1:
    plt.figure(figsize=(max(8, len(num_cols_eda)), max(6, len(num_cols_eda)-1)))
    sns.heatmap(train[num_cols_eda].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("confusion_matrices/00_correlation_heatmap.png", dpi=120)
    plt.close()
    print("[saved] 00_correlation_heatmap.png")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Preprocessing")
print("="*60)

# Detect ID column
id_col = None
for c in ["id", "Id", "ID"]:
    if c in test.columns and c != TARGET:
        id_col = c; break

test_ids = test[id_col].copy() if id_col else pd.Series(range(len(test)))

drop_train = [c for c in [id_col, TARGET] if c and c in train.columns]
X = train.drop(columns=drop_train).copy()
y = train[TARGET].copy()
X_test_raw = test.drop(columns=[id_col]).copy() if id_col and id_col in test.columns else test.copy()

print("Generating Hybrid Features (Targeting 1.000)...")

def apply_hybrid_features(df):
   
    # High score components (Triggering High Need)
    df['soil_is_dry'] = (df['Soil_Moisture'] < 25).astype(int)
    df['rain_is_low'] = (df['Rainfall_mm'] < 300).astype(int)
    df['temp_is_hot'] = (df['Temperature_C'] > 30).astype(int)
    df['wind_is_high'] = (df['Wind_Speed_kmh'] > 10).astype(int)
    
    # Low score components (Triggering Low Need)
    df['stage_is_low_need'] = df['Crop_Growth_Stage'].isin(['Harvest', 'Sowing']).astype(int) if 'Crop_Growth_Stage' in df.columns else 0
    df['is_mulched'] = (df['Mulching_Used'] == 'Yes').astype(int) if 'Mulching_Used' in df.columns else 0
    
    # The "Golden Feature"
    df['Heuristic_Score'] = (
        (df['soil_is_dry'] * 2) + 
        (df['rain_is_low'] * 2) + 
        df['temp_is_hot'] + 
        df['wind_is_high'] - 
        (df['stage_is_low_need'] * 2) - 
        df['is_mulched']
    )
    
    # --- NEW: THE HARD OVERRIDE CATEGORY ---
    conditions = [
        (df['Heuristic_Score'] <= 0),
        (df['Heuristic_Score'] > 0) & (df['Heuristic_Score'] <= 3),
        (df['Heuristic_Score'] > 3)
    ]
    choices = ['Low', 'Medium', 'High']
    # Creates an 'object' column that your dynamic LabelEncoder will catch!
    df['Formula_Verdict'] = np.select(conditions, choices, default='Medium')
    
    # --- ADD OUR PREVIOUS AGRONOMY FEATURES (For safety) ---
    df['Water_Loss_Index'] = (df['Temperature_C'] * df['Sunlight_Hours']) / (df['Humidity'] + 1)
    df['Moisture_Stress'] = df['Temperature_C'] / (df['Soil_Moisture'] + 0.1)
    
    return df

X = apply_hybrid_features(X)
X_test_raw = apply_hybrid_features(X_test_raw)

# Encode target
le_target = LabelEncoder()
y_enc     = le_target.fit_transform(y)
classes   = le_target.classes_
n_classes = len(classes)
print(f"Classes ({n_classes}): {classes}")

# Encode categoricals
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical: {cat_cols}")
print(f"Numerical  : {num_cols}")

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test_raw[col]], axis=0).astype(str)
    le.fit(combined)
    X[col]          = le.transform(X[col].astype(str))
    X_test_raw[col] = le.transform(X_test_raw[col].astype(str))

# Impute missing
X.fillna(X.median(numeric_only=True), inplace=True)
X_test_raw.fillna(X_test_raw.median(numeric_only=True), inplace=True)

# (Old apply_winning_features removed since apply_hybrid_features runs before LabelEncoding)

print(f"New feature count: {X.shape[1]}")

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.20, random_state=SEED, stratify=y_enc
)
print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}")

# Scaled versions (for NB, LR, KMeans)
scaler     = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test_raw)

# ── Helpers ──────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def kfold_cv(model, Xd, yd, name):
    scores = cross_val_score(model, Xd, yd, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"  {N_FOLDS}-Fold CV → mean={scores.mean():.4f}  std={scores.std():.4f}  each={np.round(scores,4)}")
    RESULTS[name] = scores.mean()
    return scores.mean()

def save_cm(y_true, y_pred, name, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, n_classes*2), max(5, n_classes*1.5)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13)
    plt.tight_layout()
    fname = f"confusion_matrices/cm_{name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  [saved] {fname}")

def should_run(model_name):
    if MODEL_TO_SUBMIT == "best":
        return True
    if MODEL_TO_SUBMIT == "baseline":
        return False
    # KMeans auto-runs alongside Logistic Regression
    if model_name == "kmeans":
        return MODEL_TO_SUBMIT in ("kmeans", "lr")
    return MODEL_TO_SUBMIT == model_name

# ── MODEL 1 — Decision Tree ───────────────────────────────────────────────────
if should_run("dt"):
    print("\n" + "="*60 + "\n  MODEL 1 — Decision Tree\n" + "="*60)

    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                class_weight='balanced', random_state=SEED)
    kfold_cv(dt, X, y_enc, "Decision Tree")
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_val)
    print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_dt):.4f}")
    print(classification_report(y_val, y_pred_dt, target_names=classes))
    save_cm(y_val, y_pred_dt, "Decision Tree", classes)

    # Depth sweep
    print("  Tuning max_depth ...")
    best_depth, best_dt_score = 5, 0
    for depth in [3, 5, 7, 10, 15, None]:
        sc = cross_val_score(
            DecisionTreeClassifier(max_depth=depth, random_state=SEED),
            X, y_enc, cv=skf, scoring='accuracy'
        ).mean()
        flag = " ← best" if sc > best_dt_score else ""
        print(f"    depth={str(depth):>5} → {sc:.4f}{flag}")
        if sc > best_dt_score:
            best_dt_score, best_depth = sc, depth

    RESULTS["Decision Tree (tuned)"] = best_dt_score
    print(f"  Best depth: {best_depth}  (CV: {best_dt_score:.4f})")
    

# ── MODEL 2 — Naive Bayes ────────────────────────────────────────────────────
if should_run("gnb"):
    print("\n" + "="*60 + "\n  MODEL 2 — Gaussian Naive Bayes\n" + "="*60)

    gnb = GaussianNB()
    kfold_cv(gnb, X_train_sc, y_train, "Naive Bayes")
    gnb.fit(X_train_sc, y_train)
    y_pred_gnb = gnb.predict(X_val_sc)
    print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_gnb):.4f}")
    print(classification_report(y_val, y_pred_gnb, target_names=classes))
    save_cm(y_val, y_pred_gnb, "Naive Bayes", classes)

# ── MODEL 3 — Logistic Regression ────────────────────────────────────────────
if should_run("lr"):
    print("\n" + "="*60 + "\n  MODEL 3 — Logistic Regression\n" + "="*60)

    lr = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED)
    kfold_cv(lr, X_train_sc, y_train, "Logistic Regression")
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_val_sc)
    print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_lr):.4f}")
    print(classification_report(y_val, y_pred_lr, target_names=classes))
    save_cm(y_val, y_pred_lr, "Logistic Regression", classes)

    # C sweep
    print("  Tuning C ...")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        sc = cross_val_score(
            LogisticRegression(max_iter=2000, C=C, random_state=SEED),
            X_train_sc, y_train, cv=skf, scoring='accuracy'
        ).mean()
        print(f"    C={C:<8} → {sc:.4f}")

# ── MODEL 4 — K-Means as Classifier ──────────────────────────────────────────
if should_run("kmeans"):
    print("\n" + "="*60 + "\n  MODEL 4 — K-Means (used as Classifier)\n" + "="*60)
    print("  Strategy: fit k=n_classes clusters, map each cluster to majority label.")

    km = KMeans(n_clusters=n_classes, random_state=SEED, n_init=10)
    km.fit(X_train_sc)
    cluster_map = {}
    for k in range(n_classes):
        mask = km.labels_ == k
        cluster_map[k] = int(np.bincount(y_train[mask]).argmax()) if mask.sum() > 0 else 0

    y_pred_km = np.array([cluster_map[c] for c in km.predict(X_val_sc)])
    km_acc = accuracy_score(y_val, y_pred_km)
    RESULTS["K-Means Classifier"] = km_acc
    print(f"  Cluster→Label: {cluster_map}")
    print(f"  Val Accuracy : {km_acc:.4f}  (expected low — unsupervised method)")
    print(classification_report(y_val, y_pred_km, target_names=classes))
    save_cm(y_val, y_pred_km, "K-Means Classifier", classes)

# ── MODEL 5 — Random Forest ──────────────────────────────────────────────────
if should_run("rf"):
    print("\n" + "="*60 + "\n  MODEL 5 — Random Forest\n" + "="*60)

    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    kfold_cv(rf, X, y_enc, "Random Forest")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}")
    print(classification_report(y_val, y_pred_rf, target_names=classes))
    save_cm(y_val, y_pred_rf, "Random Forest", classes)

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    fi.head(20).plot(kind='bar', color='darkorange', edgecolor='black')
    plt.title("Random Forest — Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("confusion_matrices/feature_importance_rf.png", dpi=120)
    plt.close()
    print("[saved] feature_importance_rf.png")

# ── MODEL 6 — XGBoost ────────────────────────────────────────────────────────
if should_run("xgb"):
    if XGBOOST_AVAILABLE:
        print("\n" + "="*60 + "\n  MODEL 6 — XGBoost\n" + "="*60)
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
            random_state=SEED, n_jobs=-1, verbosity=0
        )
        kfold_cv(xgb, X, y_enc, "XGBoost")
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred_xgb = xgb.predict(X_val)
        print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_xgb):.4f}")
        print(classification_report(y_val, y_pred_xgb, target_names=classes))
        save_cm(y_val, y_pred_xgb, "XGBoost", classes)

# ── MODEL 7 — LightGBM ───────────────────────────────────────────────────────
if should_run("lgbm"):
    if LGBM_AVAILABLE:
        print("\n" + "="*60 + "\n  MODEL 7 — LightGBM\n" + "="*60)
        lgbm = LGBMClassifier(
            n_estimators=1250, learning_rate=0.04, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=SEED, n_jobs=-1, verbose=-1
        )
        kfold_cv(lgbm, X, y_enc, "LightGBM")
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_val)
        print(f"  Val Accuracy: {accuracy_score(y_val, y_pred_lgbm):.4f}")
        print(classification_report(y_val, y_pred_lgbm, target_names=classes))
        save_cm(y_val, y_pred_lgbm, "LightGBM", classes)

# ── LOOCV ────────────────────────────────────────────────────────────────────
if should_run("dt"):
    print("\n" + "="*60 + "\n  LOOCV — Leave-One-Out (sample of 500)\n" + "="*60)

    # Use tuned best_depth if available, else default to 10
    _loocv_depth = best_depth if 'best_depth' in dir() else 10

    LOOCV_N = min(500, len(X_train))
    idx     = np.random.choice(len(X_train), LOOCV_N, replace=False)
    Xl, yl  = X_train.iloc[idx], y_train[idx]

    loo_preds, loo_true = [], []
    dt_loo = DecisionTreeClassifier(max_depth=_loocv_depth, random_state=SEED)
    for tr, te in LeaveOneOut().split(Xl):
        dt_loo.fit(Xl.iloc[tr], yl[tr])
        loo_preds.append(dt_loo.predict(Xl.iloc[te])[0])
        loo_true.append(yl[te][0])

    loocv_acc = accuracy_score(loo_true, loo_preds)
    RESULTS["Decision Tree (LOOCV)"] = loocv_acc
    print(f"  LOOCV Accuracy (DT, n={LOOCV_N}): {loocv_acc:.4f}")

# ── Summary Plot ──────────────────────────────────────────────────────────────
if RESULTS:
    print("\n" + "="*60 + "\n  RESULTS SUMMARY\n" + "="*60)
    res_df = pd.DataFrame.from_dict(RESULTS, orient='index', columns=['CV/Val Accuracy'])
    res_df = res_df.sort_values('CV/Val Accuracy', ascending=False)
    print(res_df.to_string())

    plt.figure(figsize=(11,5))
    colors = ['gold' if i==0 else 'steelblue' for i in range(len(res_df))]
    res_df['CV/Val Accuracy'].plot(kind='barh', color=colors[::-1], edgecolor='black')
    plt.axvline(res_df['CV/Val Accuracy'].max(), color='red', ls='--', lw=1.5, label='Best')
    plt.xlabel("Accuracy"); plt.title("Model Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig("confusion_matrices/model_comparison.png", dpi=120)
    plt.close()
    print("[saved] model_comparison.png")
else:
    print("\n[INFO] No models were run — skipping summary.")
    res_df = pd.DataFrame(columns=['CV/Val Accuracy'])

# ── Optuna Study (Optional) ──────────────────────────────────────────────────
study = None
if MODEL_TO_SUBMIT == "optuna_lgbm" and OPTUNA_AVAILABLE and LGBM_AVAILABLE:
    print("\n" + "="*60 + "\n  OPTUNA — Tuning LightGBM\n" + "="*60)
    
    def objective(trial):
        param = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': SEED,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'n_estimators': 1500 # Faster study
        }
        model = LGBMClassifier(**param)
        return cross_val_score(model, X, y_enc, cv=skf, scoring='accuracy', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # 30 trials for a quick optimization
    
    print(f"\n🚀 Best Accuracy: {study.best_value:.4f}")
    print(f"📌 Best Params: {study.best_params}")

# ── Differential Evolution Study (Optional) ──────────────────────────────────
de_result = None
if MODEL_TO_SUBMIT == "de_lgbm" and LGBM_AVAILABLE:
    from scipy.optimize import differential_evolution

    print("\n" + "="*60 + "\n  DIFFERENTIAL EVOLUTION — Tuning LightGBM\n" + "="*60)

    def de_objective(params):
        lr, n_est, leaves, child_samples = params
        model = LGBMClassifier(
            learning_rate=lr,
            n_estimators=int(n_est),
            num_leaves=int(leaves),
            min_child_samples=int(child_samples),
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        )
        score = cross_val_score(model, X, y_enc, cv=3, scoring='accuracy', n_jobs=-1).mean()
        return 1 - score  # DE minimizes, so we invert accuracy

    bounds = [
        (0.01, 0.1),   # learning_rate
        (400, 1000),   # n_estimators
        (31, 127),     # num_leaves
        (20, 500)      # min_child_samples
    ]

    print("🧬 Starting Differential Evolution Optimization...")
    de_result = differential_evolution(
        de_objective, bounds,
        strategy='best1bin', popsize=10,
        mutation=(0.5, 1), recombination=0.7,
        seed=SEED, tol=0.001, disp=True
    )

    print(f"\n🏆 Best Accuracy Found: {1 - de_result.fun:.4f}")
    print(f"📌 Best Parameters: lr={de_result.x[0]:.4f}, n_est={int(de_result.x[1])}, leaves={int(de_result.x[2])}, min_child={int(de_result.x[3])}")

# ── Final Training on Full Data ───────────────────────────────────────────────
print("\n" + "="*60 + "\n  FINAL — Train on ALL data, generate submission\n" + "="*60)

print(f"Selected model → {MODEL_TO_SUBMIT}")

# ===== MODEL SELECTION =====
if MODEL_TO_SUBMIT == "dt":
    final = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                   class_weight='balanced', random_state=SEED)

elif MODEL_TO_SUBMIT == "gnb":
    final = GaussianNB()
    X_final = scaler.fit_transform(X)
    X_test_final = scaler.transform(X_test_raw)

elif MODEL_TO_SUBMIT == "lr":
    final = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    X_final = scaler.fit_transform(X)
    X_test_final = scaler.transform(X_test_raw)

elif MODEL_TO_SUBMIT == "rf":
    final = RandomForestClassifier(n_estimators=300, max_depth=15,
                                   class_weight='balanced', random_state=SEED, n_jobs=-1)

elif MODEL_TO_SUBMIT == "xgb" and XGBOOST_AVAILABLE:
    final = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1
    )

elif MODEL_TO_SUBMIT == "lgbm" and LGBM_AVAILABLE:
    final = LGBMClassifier(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1
    )

elif MODEL_TO_SUBMIT == "blend":
    from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
    
    # 1. Our 0.967 Champion
    clf1 = LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63, 
                          random_state=SEED, n_jobs=-1)
    
    # 2. The Robust Backup (Simple but sturdy)
    clf2 = ExtraTreesClassifier(n_estimators=500, max_depth=15, 
                                 class_weight='balanced', random_state=SEED, n_jobs=-1)

    # 3. The Blend (70% trust in LGBM, 30% in Extra Trees)
    final = VotingClassifier(
        estimators=[('lgbm', clf1), ('et', clf2)],
        voting='soft', 
        weights=[0.7, 0.3]
    )
    print("🚀 Training Weighted Blend (70% LGBM / 30% Extra Trees)...")
elif MODEL_TO_SUBMIT == "triple_threat":
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier

    # 1. LightGBM (Our high-precision sniper)
    clf1 = LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63, random_state=SEED)

    # 2. XGBoost (The stabilizer)
    clf2 = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=SEED)

    # 3. CatBoost (The Category King)
    clf3 = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0, random_seed=SEED)

    # 4. The 0.97+ Weighted Blend
    final = VotingClassifier(
        estimators=[('lgbm', clf1), ('xgb', clf2), ('cat', clf3)],
        voting='soft',
        weights=[0.4, 0.3, 0.3] # Give LGBM slightly more trust
    )
    print("🚀 Training Triple Threat Ensemble (LGB x XGB x CAT)...")
elif MODEL_TO_SUBMIT == "champion" and LGBM_AVAILABLE:
    # --- THE 0.9845 CHAMPION PARAMETERS ---
    final_best_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_jobs': -1,
        'learning_rate': 0.015168742390969533,
        'num_leaves': 103,
        'max_depth': 15,
        'min_child_samples': 31,
        'feature_fraction': 0.40930497200582866,
        'bagging_fraction': 0.4586262516401655,
        'bagging_freq': 7,
        'lambda_l1': 1.9058757520549223,
        'lambda_l2': 4.208649514385759,
        'n_estimators': 2500 
    }
    final = LGBMClassifier(**final_best_params)
    print("🚀 Training the 0.9845 Master Model on FULL dataset...")

elif MODEL_TO_SUBMIT == "hybrid" and LGBM_AVAILABLE:
    # --- HYBRID FEATURE MODEL ---
    print("🚀 Training the Denoised Hybrid Model on FULL dataset...")
    final = LGBMClassifier(
        n_estimators=500,          # Dropped to 500 since it learns so fast
        learning_rate=0.05,
        max_depth=6,               
        min_child_samples=500,     
        lambda_l2=10.0,            
        cat_smooth=100,            
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1               # <--- THIS SILENCES THE WARNINGS
    )

elif MODEL_TO_SUBMIT == "optuna_lgbm" and LGBM_AVAILABLE:
    # --- STEP 3: OPTUNA-OPTIMIZED HYPERPARAMETERS ---
    if study:
        best_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': SEED,
            'n_jobs': -1,
            'learning_rate': study.best_params['learning_rate'],
            'num_leaves': study.best_params['num_leaves'],
            'max_depth': study.best_params['max_depth'],
            'min_child_samples': study.best_params['min_child_samples'],
            'feature_fraction': study.best_params['feature_fraction'],
            'bagging_fraction': study.best_params['bagging_fraction'],
            'bagging_freq': study.best_params['bagging_freq'],
            'lambda_l1': study.best_params['lambda_l1'],
            'lambda_l2': study.best_params['lambda_l2'],
            'n_estimators': 2000, # Increased for the final full-data run
        }
    else:
        # Fallback to high-performing defaults if study wasn't run
        print("[INFO] Using default high-performance parameters for LightGBM.")
        best_params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'verbosity': -1,
            'boosting_type': 'gbdt', 'random_state': SEED, 'n_jobs': -1,
            'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 12,
            'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
            'n_estimators': 2000
        }

    final = LGBMClassifier(**best_params)
    print("🚀 Training Final Optuna-Optimized Model on FULL dataset...")

elif MODEL_TO_SUBMIT == "de_lgbm" and LGBM_AVAILABLE:
    # --- DIFFERENTIAL EVOLUTION OPTIMIZED PARAMETERS ---
    if de_result is not None:
        lr, n_est, leaves, child_samples = de_result.x
        print(f"🧬 Using DE-found params: lr={lr:.4f}, n_est={int(n_est)}, leaves={int(leaves)}, min_child={int(child_samples)}")
        final = LGBMClassifier(
            learning_rate=lr,
            n_estimators=int(n_est),
            num_leaves=int(leaves),
            min_child_samples=int(child_samples),
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        # Fallback: run with strong denoising defaults
        print("[INFO] DE study not run. Using strong denoising defaults.")
        final = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=500,
            lambda_l2=10.0,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        )
    print("🚀 Training DE-Optimized Model on FULL dataset...")

elif MODEL_TO_SUBMIT == "baseline":
    # Majority class baseline
    majority_class = np.bincount(y_enc).argmax()
    preds_enc = np.full(len(X_test_raw), majority_class)
    preds_lbl = le_target.inverse_transform(preds_enc)

    submission = pd.DataFrame({
        (id_col if id_col else "id"): test_ids,
        TARGET: preds_lbl
    })
    submission.to_csv("submission.csv", index=False)
    print("Baseline submission.csv saved")
    exit()

else:
    # DEFAULT BEST MODEL
    best_name = res_df.index[0]
    print(f"Auto-selected best model: {best_name}")

    if "LightGBM" in best_name and LGBM_AVAILABLE:
        final = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8,
                               class_weight='balanced', random_state=SEED, n_jobs=-1)

    elif "XGBoost" in best_name and XGBOOST_AVAILABLE:
        final = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric='mlogloss', random_state=SEED, n_jobs=-1)

    else:
        final = RandomForestClassifier(n_estimators=300, max_depth=15,
                                       class_weight='balanced', random_state=SEED, n_jobs=-1)

# ===== TRAIN FINAL MODEL =====
if MODEL_TO_SUBMIT in ["gnb", "lr"]:
    final.fit(X_final, y_enc)
    preds_enc = final.predict(X_test_final)
else:
    final.fit(X, y_enc)
    preds_enc = final.predict(X_test_raw)

preds_lbl = le_target.inverse_transform(preds_enc)

submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    TARGET: preds_lbl
})

if MODEL_TO_SUBMIT == "hybrid":
    submission_name = "submission_hybrid.csv"
elif MODEL_TO_SUBMIT == "champion":
    submission_name = "submission_0.9845_optuna.csv"
elif MODEL_TO_SUBMIT == "optuna_lgbm":
    submission_name = "submission_optuna_final.csv"
elif MODEL_TO_SUBMIT == "de_lgbm":
    submission_name = "submission_de_lgbm.csv"
else:
    submission_name = "submission.csv"

submission.to_csv(submission_name, index=False)
print(f"✅ MISSION ACCOMPLISHED. File '{submission_name}' is ready — {submission.shape}")
