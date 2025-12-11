# 🏦 Projet Data Science : Analyse de Transactions Bancaires

Ce dépôt illustre un projet complet de Data Science appliqué à des données de transactions bancaires, en suivant la structure pédagogique du document « Correction-Projet ».

---

## 1. Contexte métier

Dans le secteur bancaire, le volume de transactions et la diversité des canaux (ATM, agence, en ligne) rendent difficile la détection manuelle des opérations à risque.  
L’objectif est de construire un pipeline de Machine Learning permettant d’explorer les transactions et de simuler une détection de transactions potentiellement frauduleuses.

---

## 2. Données utilisées

- **Fichier principal :** `bank_transactions_data.csv`  
- **Granularité :** 1 ligne = 1 transaction  
- **Types de variables (exemples) :**
  - Identifiants : `TransactionID`, `AccountID`
  - Financières : `TransactionAmount`, `AccountBalance`
  - Temporelles : `TransactionDate`, `PreviousTransactionDate`
  - Comportement : `Channel`, `Location`, `DeviceID`, `LoginAttempts`, `TransactionDuration`
  - Démographiques : `CustomerAge`, `CustomerOccupation`

Une cible binaire simulée `is_risky` est construite à partir du montant de la transaction (transactions très élevées marquées comme « à risque »).  
Dans un cas réel, cette cible serait fournie par l’historique des fraudes connues.

---

## 3. Pipeline Data Science

Le code principal se trouve dans un script (ou notebook) inspiré de `PROJET_DS.ipynb` et suit les étapes suivantes :

1. **Importation des bibliothèques**  
   NumPy, Pandas, Matplotlib, Seaborn, scikit-learn (RandomForestClassifier, train_test_split, métriques).

2. **Chargement des données**  
   - Lecture de `bank_transactions_data.csv` avec Pandas.  
   - Affichage de la taille du dataset et de la liste des colonnes.

3. **Construction de la cible `is_risky` (exemple pédagogique)**  
   - Transactions dont le montant est supérieur au 95e centile marquées comme `1`.  
   - Les autres transactions marquées comme `0`.

4. **Préparation des features**  
   - Suppression des colonnes purement identifiantes (`TransactionID`, `AccountID`, dates, IP).  
   - Encodage one-hot des variables catégorielles.  
   - Séparation en `X` (features) et `y` (cible).

5. **Nettoyage et imputation**  
   - Utilisation de `SimpleImputer(strategy="mean")` pour remplacer les valeurs manquantes des colonnes numériques.  
   - Création d’une matrice propre `X_clean`.

6. **Analyse exploratoire (EDA)**  
   - Statistiques descriptives sur les montants et soldes.  
   - Histogramme de la distribution de `TransactionAmount`.  
   - Possibilité d’ajouter une heatmap de corrélation sur un sous-ensemble de variables.

7. **Split Train / Test**  
   - `train_test_split` avec `test_size=0.2`, `random_state=42`, `stratify=y`.  
   - Objectif : évaluer la capacité de généralisation du modèle.

8. **Modélisation : Random Forest**  
   - Utilisation de `RandomForestClassifier(n_estimators=100, class_weight="balanced")`.  
   - Entraînement sur le jeu d’entraînement uniquement.

9. **Évaluation**  
   - Calcul de l’accuracy.  
   - Rapport de classification (precision, recall, f1-score).  
   - Matrice de confusion visualisée via Seaborn.

---

## 4. Résultats et interprétation

- Le modèle permet d’identifier une partie des transactions marquées comme `is_risky` sur le jeu de test.  
- L’accuracy est complétée par l’analyse de la **precision** et du **recall** sur la classe `1` (transactions à risque).  
- Dans un contexte bancaire réel, la priorité serait de maximiser le recall de la fraude tout en contrôlant le nombre de faux positifs.

---

## 5. Limites et pistes d’amélioration

- La cible `is_risky` est ici simulée à partir d’un simple seuil de montant, ce qui ne reflète pas toute la complexité de la fraude réelle.  
- Le modèle pourrait être amélioré par :
  - L’ingénierie de features (fréquence des transactions par client, temps depuis la dernière transaction, agrégations par canal, etc.).  
  - L’utilisation de méthodes dédiées aux données déséquilibrées (SMOTE, ajustement de seuils de décision, etc.).  
  - La mise en place d’une validation croisée plus rigoureuse.

---

## 6. Utilisation

1. Cloner le dépôt.  
2. Placer `bank_transactions_data.csv` à la racine du projet.  
3. Exécuter le script Python principal ou ouvrir le notebook correspondant.  
4. Consulter les sorties (métriques, graphiques) pour analyser les performances du modèle.

---

## 7. Références

Ce projet suit la logique pédagogique du document « Correction-Projet : Anatomie d’un projet Data Science » (contexte métier, data wrangling, EDA, split, modélisation, évaluation).
