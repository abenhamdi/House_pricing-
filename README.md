# 🏠 House Pricing Prediction

Un projet de machine learning visant à prédire les prix de maisons à partir de diverses caractéristiques. Il s'agit d'une approche supervisée utilisant des modèles de régression sur un dataset structuré.

## 🎯 Objectif

L'objectif principal est de :

* Nettoyer et explorer un jeu de données immobilier,
* Préparer les données pour l'entraînement (feature engineering, encodage, normalisation...),
* Entraîner plusieurs modèles de régression,
* Évaluer et comparer leurs performances,
* Prédire les prix de maisons avec précision.

## 🛠️ Technologies & Librairies

* Python 3.x
* `pandas`, `numpy` – manipulation des données
* `matplotlib`, `seaborn` – visualisation
* `scikit-learn` – modélisation, évaluation
* `xgboost` (optionnel) – modèle avancé de boosting

## 📁 Structure du projet

```
House_pricing-/
│
├── house-pricing.ipynb       # Notebook principal contenant tout le pipeline ML
├── train.csv                 # Données d'entraînement (features + target)
├── test.csv                  # Données de test à prédire (features uniquement)
└── submission.csv            # Exemple de fichier de soumission Kaggle
```

## 📊 Étapes principales du notebook

1. **Chargement et exploration des données**

   * Analyse des types de variables, valeurs manquantes, outliers
2. **Prétraitement**

   * Imputation, encodage des variables catégorielles
   * Normalisation et transformation
3. **Modélisation**

   * Régression linéaire
   * RandomForestRegressor
   * Gradient Boosting / XGBoost (si utilisé)
4. **Évaluation**

   * RMSE, MAE
   * Courbes d’erreur et importance des variables
5. **Soumission**

   * Génération du fichier `submission.csv`

## 🧠 Résultats & Insights

* Importance des variables : surface habitable, quartier, qualité générale, etc.
* Feature engineering significatif pour améliorer la précision du modèle.
* Utilisation de cross-validation pour éviter l'overfitting.

## 🚀 Lancer le projet

1. Cloner le repo :

```bash
git clone https://github.com/abenhamdi/House_pricing-.git
cd House_pricing-
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
# ou manuellement :
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

3. Lancer Jupyter :

```bash
jupyter notebook
```

4. Ouvrir `house-pricing.ipynb` et exécuter les cellules.

## 📝 TODO & améliorations possibles

* Ajouter des modèles avancés (LightGBM, CatBoost)
* Hyperparamétrage automatique (GridSearchCV, Optuna)
* Déploiement avec Flask ou Streamlit
* Dashboard interactif avec visualisations

## 📚 Source des données

Le dataset utilisé provient probablement de la compétition [**Kaggle House Prices: Advanced Regression Techniques**](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## 👤 Auteur

* **abenhamdi** – [@abenhamdi](https://github.com/abenhamdi)
