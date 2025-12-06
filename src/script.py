import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import shap



# =====================================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================================
df = pd.read_csv(r"C:\Users\hp\Documents\ML\bank\bank-additional-full.csv", sep=';')

df = df.drop_duplicates()
df = df.replace("unknown", "Unknown")

X = df.drop("y", axis=1)
y = df["y"].replace({"yes": 1, "no": 0})


# =====================================================================
# 2. PREPROCESSING
# =====================================================================
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# =====================================================================
# 3. MODELES DE BASE
# =====================================================================

# Logistic Regression
log_reg = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])

log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Random Forest
rf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    ))
])

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost
xgb = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ))
])

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


# =====================================================================
# 4. SCORES INITIAUX
# =====================================================================
print("\n=== SCORES INITIAUX ===")
models_initial = {
    "Logistic Regression": (y_pred_lr, log_reg),
    "Random Forest": (y_pred_rf, rf),
    "XGBoost": (y_pred_xgb, xgb)
}

for name, (ypred, model) in models_initial.items():
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, ypred)
    print(f"{name} → F1-score = {f1:.3f}, AUC = {auc:.3f}")


print("\n###################################################################")
print("### HYPERPARAMETER TUNING")
print("###################################################################\n")


# =====================================================================
# 5. GRIDSEARCH : LOGISTIC REGRESSION
# =====================================================================
param_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs', 'liblinear']
}

grid_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)
print("Best parameters LR :", grid_lr.best_params_)


# =====================================================================
# 6. RANDOMIZEDSEARCH : RANDOM FOREST (rapide & optimal)
# =====================================================================
param_dist_rf = {
    'clf__n_estimators': randint(150, 500),
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__max_features': ['sqrt', 'log2']
}

random_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist_rf,
    n_iter=25,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_rf.fit(X_train, y_train)
print("Best parameters RF :", random_rf.best_params_)


# =====================================================================
# 7. GRIDSEARCH : XGBOOST
# =====================================================================
param_xgb = {
    'clf__n_estimators': [200, 400],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.8, 1],
    'clf__colsample_bytree': [0.7, 0.9]
}

grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_xgb,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)
print("Best parameters XGB :", grid_xgb.best_params_)


# =====================================================================
# 8. COMPARAISON FINALE
# =====================================================================
print("\n\n=================== COMPARAISON FINALE ===================\n")

models_tuned = {
    "Logistic Regression": grid_lr,
    "Random Forest": random_rf,
    "XGBoost": grid_xgb
}

for name, model in models_tuned.items():
    best = model.best_estimator_
    y_pred = best.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])

    print(f"\n=== {name} ===")
    print("F1-score :", np.round(f1, 3))
    print("AUC :", np.round(auc, 3))
    print(classification_report(y_test, y_pred))

# ===============================================================
# 1. SAMPLE POUR ACCELERER SHAP (300 lignes)
# ===============================================================
X_test_sample = X_test.sample(300, random_state=42)

# apply preprocessing
X_test_sample_transformed = best_pipeline.named_steps["preprocess"].transform(X_test_sample)

# ===============================================================
# 2. RECUPERATION DES NOMS DES FEATURES APRES ONEHOT ENCODING
# ===============================================================
encoder = best_pipeline.named_steps['preprocess'].named_transformers_['cat']
ohe_features = encoder.get_feature_names_out(categorical_cols)

final_feature_names = np.concatenate([ohe_features, numeric_cols])

# =====================================================================
# 9. SHAP POUR RANDOM FOREST OPTIMISÉ
# =====================================================================

print("\n==================== SHAP EXPLANATION ====================\n")

# 🔹 1. Récupération du meilleur pipeline Random Forest
best_pipeline = random_rf.best_estimator_

# 🔹 2. Extraction du vrai modèle Random Forest (le step 'clf')
rf_model = best_pipeline.named_steps["clf"]

# 🔹 3. Sample pour accélérer SHAP
X_test_sample = X_test.sample(300, random_state=42)

# 🔹 4. Transformation des données avec le bon preprocess
X_test_sample_transformed = best_pipeline.named_steps["preprocess"].transform(X_test_sample)

# 🔹 5. Récupération des noms finaux des features
encoder = best_pipeline.named_steps['preprocess'].named_transformers_['cat']
ohe_features = encoder.get_feature_names_out(categorical_cols)
final_feature_names = np.concatenate([ohe_features, numeric_cols])

# 🔹 6. Explainer optimisé pour RandomForest
explainer = shap.TreeExplainer(
    rf_model,
    model_output="probability",
    feature_perturbation="tree_path_dependent"
)

# 🔹 7. Calcul rapide des valeurs SHAP
shap_values = explainer.shap_values(X_test_sample_transformed)

# 🔹 8. Summary Plot
shap.summary_plot(
    shap_values[1],  # classe positive
    X_test_sample_transformed,
    feature_names=final_feature_names
)
