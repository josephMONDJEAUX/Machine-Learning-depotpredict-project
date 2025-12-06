# Machine-Learning-depotpredict-project
Ce projet consiste à prédire les clients succeptibles de souscrire à un compte épargne bloqué dans une banque basé sur des données marketing

## Objectif
Prédire la variable cible `y` (yes/no) indiquant si un client accepte une offre de dépôt à terme lors d'une campagne de marketing direct.


##  Workflow du Projet

### 1. Analyse Exploratoire (EDA)
- Statistiques descriptives
- Corrélation entre variables
- Distribution des features clés
- Analyse du déséquilibre des classes

### 2. Préparation des données
- Encodage catégoriel via OneHotEncoder
- Standardisation des variables numériques
- Pipeline automatique avec `ColumnTransformer`
- Split train/test 

### 3. Entraînement de 3 modèles
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

### 4. Optimisation d’hyperparamètres
- `GridSearchCV` pour LogReg & XGBoost
- `RandomizedSearchCV` pour RandomForest
- Optimisation ciblée du F1-score (classes déséquilibrées)

### 5. Interprétabilité SHAP
- SHAP TreeExplainer sur échantillon test
- Summary plot des variables influentes
- Compatible avec pipeline scikit-learn


## Technologies utilisées
- Python 3
- Pandas / Numpy
- Matplotlib / Seaborn
- Scikit-Learn
- XGBoost
- SHAP


## Résultats
- **Meilleur modèle : RandomForest optimisé**
- Score F1 : > 0.80
- AUC : > 0.92
- Insights via SHAP : liés à la faiblesse de ma machine, je completerais cette partie
  


## Dataset
Dataset original provenant d’une banque portugaise :  
UCI Machine Learning Repository — Bank Marketing Dataset.


## Auteur
Joseph MONDJEAUX

Projet réalisé en 2025 — Data Science, Machine Learning & Interprétabilité.
