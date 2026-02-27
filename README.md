# Projet TP Stats 2 — Classification Forest Cover (MLP vs MLP + SMOA)

## 1) Description du projet
Ce projet a pour objectif de **prédire le type de couverture forestière** (`Cover_Type`) à partir des variables du dataset Covertype, puis de **comparer deux approches** :

- **MLP baseline** (réseau de neurones multicouche avec hyperparamètres fixes)
- **MLP optimisé par SMOA** (optimiseur métaheuristique appliqué aux hyperparamètres du MLP)

Le but principal est de **tester l’efficacité d’un optimiseur métaheuristique** (SMOA) par rapport à un entraînement MLP standard.

---

## 2) Données utilisées
- Fichier principal : `data/covtype.csv`
- Variable cible : `Cover_Type`
- Nature des variables : caractéristiques topographiques, distances, types de sols (`Soil_Type*`) et zones sauvages (`Wilderness_Area*`).

### Prétraitement
Selon le notebook/script :
- Séparation `X / y`
- Normalisation via `StandardScaler`
- Dans la baseline (`mlp_smoa.ipynb`), regroupement des colonnes one-hot :
  - `Soil_Type1..40` → une colonne `Soil_Type`
  - `Wilderness_Area1..4` → une colonne `Wilderness_Area`
- Découpage train/test (stratifié, `test_size=0.2`)

---

## 3) Algorithmes utilisés

### A. MLP (Multilayer Perceptron) — Baseline
Implémenté dans `mlp_smoa.ipynb` (partie baseline MLP), avec une architecture fixe (ex. couches cachées `(64, 32)`), puis entraînement incrémental (`partial_fit`) et suivi de la courbe d’accuracy.

### B. SMOA (optimiseur métaheuristique)
Implémenté dans `mlp.ipynb` (classe `SMOA`) et utilisé pour rechercher de meilleurs hyperparamètres MLP :
- taille des couches cachées
- fonction d’activation
- solveur
- `alpha`, `learning_rate_init`, `tol`, etc.

SMOA combine :
- un mouvement vers le meilleur candidat courant,
- une recherche locale,
- une adaptation des paramètres de recherche,
- un arrêt anticipé en absence d’amélioration.

### C. MLP + SMOA
Une fois les meilleurs hyperparamètres trouvés par SMOA, un MLP final est entraîné et évalué sur le test set.

---

## 4) Résultats disponibles

### Dossier baseline
- `results_mlp/classification_report_mlp.txt`
- (potentiellement) modèle et graphes baseline (`mlp_baseline.pkl`, `accuracy_mlp.png`, etc.)

### Dossier SMOA
- `results_smoa/best_hyperparameters.json`
- `results_smoa/classification_report_smoa.txt`
- (potentiellement) modèle/scaler/graphes SMOA

### Indicateurs observés (fichiers actuels)
- Baseline MLP : accuracy globale ≈ **0.83**
- MLP + SMOA : accuracy globale ≈ **0.90**

> Remarque : les rapports montrent des codages de classes différents selon le pipeline (`1..7` vs `0..6`), ce qui doit être gardé en tête lors de la comparaison détaillée classe par classe.

---

## 5) Interprétation du but scientifique
Le projet répond à la question suivante :

**Un optimiseur métaheuristique (SMOA) améliore-t-il les performances d’un MLP sur la classification Forest Cover ?**

Au vu des résultats sauvegardés, l’approche **MLP + SMOA** obtient une meilleure performance globale que la baseline MLP, ce qui soutient l’intérêt de l’optimisation métaheuristique pour le réglage des hyperparamètres.

---

## 6) Structure du projet

- `app.py` : application Streamlit pour visualiser les données, comparer les modèles, afficher les rapports et faire des prédictions.
- `mlp_smoa.ipynb` : notebook baseline MLP (prétraitement + entraînement + évaluation + sauvegarde).
- `mlp.ipynb` : notebook contenant l’approche SMOA + MLP.
- `data/` : dataset (`covtype.csv`).
- `results_mlp/` : sorties baseline.
- `results_smoa/` : sorties SMOA.

---

## 7) Lancer le projet

### A. Entraîner / générer les résultats (notebooks)
Ouvrir et exécuter :
- `mlp_smoa.ipynb` (baseline)
- `mlp.ipynb` (SMOA + MLP)

### B. Lancer l’interface de comparaison
```bash
streamlit run app.py
```

---

## 8) Conclusion
Ce TP met en évidence l’impact du **choix des hyperparamètres** sur les performances d’un réseau de neurones. L’utilisation d’un optimiseur métaheuristique comme **SMOA** semble améliorer significativement la qualité de classification par rapport à une configuration MLP fixe.