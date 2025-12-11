1. Le Contexte Métier et la Mission
🎯 Le Problème (Business Case)

La fraude bancaire est un défi majeur pour les banques, fintechs et plateformes de paiement.
Chaque transaction frauduleuse entraîne :

Une perte financière directe

Un risque juridique

Une perte de confiance des clients

À l’inverse, bloquer une transaction légitime génère :

Frustration du client

Appels coûteux au service client

⚠️ Enjeu critique : la matrice des coûts d’erreur

Comme dans le projet médical de référence :

Faux Positif (FP) : Transaction normale bloquée → mécontentement

Faux Négatif (FN) : Fraude non détectée → perte financière importante

👉 Le Recall est la métrique la plus importante
On préfère alerter trop que rater une fraude.

2. Les Données (L’Input)

Le dataset Bank Transactions Data contient des transactions bancaires complètes :
comportement client, données techniques, informations temporelles et financières.

🧩 Les colonnes principales

TransactionAmount

TransactionType (Debit / Credit)

Location

DeviceID

IP Address

Channel (POS, ATM, Online…)

CustomerAge

CustomerOccupation

TransactionDuration

LoginAttempts

AccountBalance

TransactionDate, PreviousTransactionDate

🎯 Variable cible (y)

Le dataset ne contient pas de colonne "fraud".
➡️ Le projet se concentre donc sur la détection d’anomalies (unsupervised learning).

3. Le Code Python (Laboratoire)

Cette section reprend la structure du fichier Correction Projet.md :
➡️ uniquement les extraits indispensables du code, accompagnés d’explications pédagogiques.

--- PHASE 1 : ACQUISITION & STRUCTURATION ---
import pandas as pd
import numpy as np

df = pd.read_csv("bank_transactions_data.csv")
df.head()


Objectif :

Charger le dataset

Vérifier les premières lignes pour comprendre les variables

--- PHASE 2 : NETTOYAGE (DATA WRANGLING) ---
Problème du NaN

Comme dans Correction Projet.md :

Les algorithmes de Machine Learning ne tolèrent pas les valeurs manquantes.

Imputation des colonnes numériques
from sklearn.impute import SimpleImputer

num_cols = ["TransactionAmount", "TransactionDuration", "AccountBalance"]
imputer_num = SimpleImputer(strategy="mean")

df[num_cols] = imputer_num.fit_transform(df[num_cols])


Explication :

fit() calcule la moyenne pour chaque colonne

transform() remplace les trous

⚠️ Data Leakage

La bonne pratique :

Split data

Fit uniquement sur le Train

Transformer le Test

Encodage des variables catégorielles
df_encoded = pd.get_dummies(
    df,
    columns=["TransactionType", "Location", "Channel", "CustomerOccupation"],
    drop_first=True
)


Pourquoi ?
Les algorithmes ne comprennent pas le texte (Houston, Credit, Online…).
On crée des colonnes binaires (0/1).

--- PHASE 3 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
from sklearn.model_selection import train_test_split

X = df_encoded.drop("TransactionID", axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

Pourquoi 80/20 ?

80% pour apprendre les comportements

20% pour valider les résultats

Pourquoi random_state=42 ?

Reproductibilité scientifique (comme dans le projet médical).

--- PHASE 4 : INTELLIGENCE ARTIFICIELLE (ANOMALY DETECTION) ---

Le dataset n’ayant pas de label, on applique un modèle non supervisé.

Isolation Forest (équivalent du Random Forest pour anomalies)
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X_train)

df["anomaly_score"] = model.decision_function(X)
df["is_fraud"] = model.predict(X)

Comment interpréter ?

decision_function() → score d’anomalie

predict() :

1 → transaction normale

-1 → transaction suspecte

4. Analyse Exploratoire (EDA)

Comme dans le fichier Correction Projet.md, l'objectif est de comprendre le comportement des données.

📊 Points clés à analyser

Distribution des montants

Transactions nocturnes

Localisation incohérente

Nombre de tentatives de login

Montants anormaux par âge

Appareil utilisé (DeviceID inhabituel)

5. FOCUS THÉORIQUE : Pourquoi Isolation Forest ?

Comme expliqué dans le document de référence concernant Random Forest :

A. La faiblesse de l’arbre individuel

Un arbre seul apprend trop les cas extrêmes → haute variance.

B. La force de la forêt

Isolation Forest crée une forêt d’arbres aléatoires.
Les anomalies sont isolées en peu de divisions → elles ressortent naturellement.

C. Avantages

Rapide

Robuste

Non linéaire

Insensible aux distributions non normales

6. Évaluation (L’Heure de Vérité)

Si un label fraude existait, on évaluerait les performances via :

A. Matrice de Confusion

TP : fraudes détectées

TN : normales détectées

FP : faux blocages clients

FN : fraudes non détectées

B. Métriques

Precision → éviter les fausses alertes

Recall → attraper toutes les fraudes

F1-score → bilan global

⚠️ En fraude bancaire :
➡️ Le Recall est prioritaire (on ne veut jamais rater une fraude).

7. Conclusion du Projet

Ce projet est parfaitement aligné avec la méthodologie exposée dans le fichier Correction Projet.md :

Compréhension métier avant tout

Nettoyage des données indispensable

Encodage réfléchi

Split bien réalisé pour éviter les fuites de données

Choix du modèle en fonction du contexte métier

Priorité au Recall dans l’évaluation

🚀 Ce dataset est idéal pour :

Détection d’anomalies

Profilage comportemental

Systèmes d’alerte en temps réel

Construction d’un moteur antifraude
