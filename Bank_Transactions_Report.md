# Ghada chakhtoune
#  24010409


<img src="https://github.com/user-attachments/assets/9606d462-735a-401a-8d8f-a15fd7cab70f" width="200">

# ANATOMIE D'UN PROJET DATA SCIENCE

## *Application au Dataset : Bank Transactions Data*

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)

La fraude bancaire est un défi majeur pour les banques, fintechs et
plateformes de paiement.\
Chaque transaction frauduleuse entraîne : - Une **perte financière
directe** - Un **risque juridique** - Une **perte de confiance** des
clients

À l'inverse, bloquer une transaction légitime génère : - Frustration du
client\
- Appels coûteux au service client

### Enjeu critique : la matrice des coûts d'erreur

-   **Faux Positif (FP)** : Transaction normale bloquée →
    mécontentement
-   **Faux Négatif (FN)** : Fraude non détectée → perte financière
    importante

**Le Recall est la métrique la plus importante**

------------------------------------------------------------------------

## 2. Les Données (L'Input)

### Colonnes principales

-   TransactionAmount
-   TransactionType
-   Location
-   DeviceID
-   IP Address
-   Channel
-   CustomerAge
-   CustomerOccupation
-   TransactionDuration
-   LoginAttempts
-   AccountBalance
-   TransactionDate
-   PreviousTransactionDate

### Cible

Pas de colonne "fraud" → apprentissage non supervisé.

------------------------------------------------------------------------

## 3. Le Code Python (Laboratoire)

### --- PHASE 1 : ACQUISITION ---

``` python
import pandas as pd
df = pd.read_csv("bank_transactions_data.csv")
df.head()
```

------------------------------------------------------------------------

### --- PHASE 2 : NETTOYAGE ---

``` python
from sklearn.impute import SimpleImputer

num_cols = ["TransactionAmount", "TransactionDuration", "AccountBalance"]
imputer_num = SimpleImputer(strategy="mean")
df[num_cols] = imputer_num.fit_transform(df[num_cols])
```

------------------------------------------------------------------------

### --- PHASE 3 : ENCODAGE ---

``` python
df_encoded = pd.get_dummies(
    df,
    columns=["TransactionType", "Location", "Channel", "CustomerOccupation"],
    drop_first=True
)
```

------------------------------------------------------------------------

### --- PHASE 4 : SPLIT ---

``` python
from sklearn.model_selection import train_test_split

X = df_encoded.drop("TransactionID", axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

------------------------------------------------------------------------

## --- PHASE 5 : DÉTECTION D'ANOMALIES ---

``` python
from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
model.fit(X_train)

df["anomaly_score"] = model.decision_function(X)
df["is_fraud"] = model.predict(X)
```

------------------------------------------------------------------------

## 4. Analyse Exploratoire (EDA)

-   Distribution des montants
-   Transactions nocturnes
-   Localisation incohérente    
-   Tentatives de login élevées
-   Appareil inhabituel

------------------------------------------------------------------------

## 5. Conclusion

Ce dataset est idéal pour :
- la détection d'anomalies
- le scoring de risque
- un moteur antifraude
