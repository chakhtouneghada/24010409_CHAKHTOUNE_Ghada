1. Le Contexte Métier et la Mission
Le Problème (Business Case)
Dans le secteur bancaire, le volume élevé de transactions et la diversité des canaux (agences, distributeurs automatiques, services en ligne) rendent difficile la détection manuelle des opérations anormales ou potentiellement frauduleuses.
Certaines transactions présentent des montants inhabituels ou des caractéristiques atypiques (contexte, fréquence, profil client) et peuvent représenter un risque, mais restent noyées dans un flux massif d’opérations.

Objectif : Mettre en place un pipeline de Machine Learning capable de distinguer des transactions « normales » de transactions « à risque », en s’appuyant sur les informations disponibles dans un fichier de transactions bancaires.
L’Enjeu critique : La matrice des coûts d’erreur est asymétrique.

Classer comme « à risque » une transaction légitime (Faux Positif) peut générer du stress pour le client, des vérifications manuelles et des coûts opérationnels.

Classer comme « normale » une transaction réellement problématique (Faux Négatif) peut entraîner des pertes financières, une fraude non détectée et des risques de non‑conformité.

Dans un contexte de détection de risque, il est donc important de prioriser le rappel (Recall) sur la classe « à risque », quitte à accepter davantage de Faux Positifs.

Les Données (L’Input)
Nous utilisons un fichier de transactions bancaires : bank_transactions_data.csv.

X (Features) : ce sont les caractéristiques descriptives de chaque transaction, comprenant par exemple :

Identifiants techniques : TransactionID, AccountID

Variables financières : TransactionAmount, AccountBalance

Informations temporelles : TransactionDate, PreviousTransactionDate

Contexte de la transaction : Location, Channel, DeviceID, IP Address

Profil client : CustomerAge, CustomerOccupation, LoginAttempts, TransactionDuration

y (Target) : une cible binaire is_risky est construite de manière pédagogique.

0 = Transaction considérée « normale »

1 = Transaction considérée « à risque potentielle » (par exemple les 5% de plus gros montants)

2. Le Code Python (Laboratoire)
Ce script est la paillasse de laboratoire. Il contient toutes les manipulations nécessaires : chargement des données, construction de la cible, nettoyage, analyse exploratoire, séparation Train/Test, entraînement d’un modèle Random Forest et audit de ses performances.

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")

# --- PHASE 1 : ACQUISITION DES DONNÉES BANCAIRES ---
df = pd.read_csv("bank_transactions_data.csv")

print("=== APERÇU DU DATASET ===")
print(f"Taille du dataset : {df.shape}")
print("Colonnes disponibles :")
print(df.columns.tolist())
print()

# Construction d'une cible binaire pédagogique : is_risky
threshold = df["TransactionAmount"].quantile(0.95)
df["is_risky"] = (df["TransactionAmount"] > threshold).astype(int)

print("Colonne cible créée : 'is_risky' (0 = normal, 1 = transaction à risque potentielle)")
print(f"Répartition de la cible :\n{df['is_risky'].value_counts(normalize=True)}\n")

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE & PRÉPARATION) ---
cols_to_drop = [
    "TransactionID",
    "AccountID",
    "TransactionDate",
    "PreviousTransactionDate",
    "IP Address"
]
df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

X = df_model.drop("is_risky", axis=1)
y = df_model["is_risky"]

# Encodage des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Nombre total de valeurs manquantes restantes : {X_clean.isnull().sum().sum()}\n")

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Statistiques Descriptives (variables financières) ---")
num_cols = [c for c in X_clean.columns if "TransactionAmount" in c or "AccountBalance" in c]
if len(num_cols) > 0:
    print(X_clean[num_cols].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["TransactionAmount"], kde=True)
plt.title("Distribution du montant des transactions")
plt.xlabel("TransactionAmount")
plt.tight_layout()
plt.show()

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {X_train.shape[0]} échantillons")
print(f"Test        : {X_test.shape[0]} échantillons\n")

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST) ---
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

print(f"\n--- Accuracy Globale : {accuracy_score(y_test, y_pred)*100:.2f}% ---")
print("\n--- Rapport Détaillé ---")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion : Transactions normales vs à risque')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.show()
3. Analyse Approfondie : Nettoyage (Data Wrangling)
Le Problème Mathématique du « Vide »
Les algorithmes de Machine Learning reposent sur des opérations d’algèbre linéaire qui ne supportent pas la présence de valeurs NaN.
Une seule valeur manquante dans une colonne peut suffire à faire échouer un entraînement ou une prédiction.

Dans un jeu de données de transactions bancaires, ces valeurs manquantes peuvent provenir d’informations client incomplètes, de champs facultatifs non renseignés ou d’erreurs d’enregistrement.
Il est donc indispensable de remplacer ou de traiter ces « vides » avant de passer à la phase de modélisation.

La Mécanique de l’Imputation
Nous utilisons SimpleImputer(strategy="mean").

Apprentissage (fit) : l’imputer parcourt chaque colonne numérique de X et calcule la moyenne des valeurs disponibles. Il mémorise cette moyenne pour chaque feature (par exemple, le montant moyen ou le solde moyen).

Transformation (transform) : lors de la transformation, toutes les valeurs manquantes d’une colonne sont remplacées par la moyenne apprise.

Cette stratégie produit un tableau X_clean sans NaN, prêt à être utilisé par le modèle.

💡 Coin de l’Expert (Data Leakage)
Dans un projet rigoureux, il faut éviter que l’information du jeu de test se retrouve injectée dans les statistiques utilisées pour le nettoyage.
La bonne pratique consiste à ajuster l’imputer uniquement sur le jeu d’entraînement, puis à appliquer cette transformation au jeu de test.

4. Analyse Approfondie : Exploration (EDA)
C’est l’étape de « profilage » des transactions.

Décrypter .describe()
L’appel à describe() sur des variables comme TransactionAmount ou AccountBalance fournit plusieurs informations clés :

Mean (Moyenne) vs 50% (Médiane) : si la moyenne est nettement plus élevée que la médiane, cela indique une distribution asymétrique, tirée par quelques transactions de très gros montant.

Std (Écart-type) : mesure la dispersion des valeurs. Un écart-type élevé signifie que les montants sont très variés, ce qui peut rendre le problème plus complexe pour le modèle.

La Multicollinéarité (Le problème de la redondance)
En étudiant une matrice de corrélation sur les variables numériques (montant, solde, durée, etc.), certaines paires de colonnes peuvent apparaître fortement corrélées.

Sur le plan économique, cela peut être logique : un solde courant peut être lié à un solde moyen ou à la fréquence des transactions.

Pour un Random Forest, cette redondance pose peu de problèmes car les arbres sélectionnent des sous‑ensembles de variables et gèrent bien les corrélations. Pour des modèles linéaires, une forte multicolinéarité peut rendre les coefficients difficiles à interpréter et instables.

5. Analyse Approfondie : Méthodologie (Split)
Le Concept : La Garantie de Généralisation
Le but du Machine Learning n’est pas de mémoriser les transactions passées, mais de généraliser à de nouvelles opérations.
Séparer les données en deux ensembles – un pour l’entraînement, un pour le test – permet de vérifier la capacité réelle du modèle à se comporter correctement sur des données jamais vues.

Les Paramètres sous le capot
La séparation utilisée est :

test_size=0.2 : environ 80% des transactions sont utilisées pour l’entraînement, 20% pour le test.

random_state=42 : la graine fixe le tirage aléatoire, ce qui garantit la reproductibilité des résultats.

stratify=y : le ratio entre transactions normales et à risque est conservé dans les deux sous‑ensembles.

Le ratio 80/20 permet au modèle de disposer d’assez d’exemples pour apprendre des schémas robustes tout en gardant suffisamment de données pour évaluer la performance finale de manière fiable.
La reproductibilité est essentielle pour pouvoir comparer plusieurs versions du modèle dans le temps.

6. FOCUS THÉORIQUE : L’Algorithme Random Forest 🌲
A. La Faiblesse de l’Individu (Arbre de Décision)
Un arbre de décision unique pose des questions successives sur les variables (montant, âge du client, canal, localisation, etc.) pour aboutir à une prédiction.
Le problème est qu’il peut facilement sur‑apprendre : si une transaction très atypique apparaît, l’arbre peut créer une règle très spécifique juste pour ce cas, ce qui conduit à une forte variance.

B. La Force du Groupe (Bagging)
Random Forest signifie « Forêt Aléatoire ». Le modèle construit de nombreux arbres à partir de variations des données et des variables.

Bootstrapping (Diversité des Échantillons) : chaque arbre est entraîné sur un échantillon tiré avec remise à partir des données d’entraînement. Chaque arbre voit donc une version légèrement différente de l’historique de transactions.

Feature Randomness (Diversité des Questions) : à chaque nœud, un arbre ne considère qu’un sous‑ensemble aléatoire de variables pour décider du meilleur split. Cela oblige les arbres à explorer des combinaisons de features moins évidentes et évite qu’ils ne se focalisent tous sur la même variable (par exemple, uniquement le montant).

C. Le Consensus (Vote)
Pour une nouvelle transaction, tous les arbres de la forêt produisent une prédiction (normale ou à risque).
La classe finale est déterminée par un vote majoritaire. Les erreurs individuelles de certains arbres se compensent, ce qui renforce la stabilité du modèle et la qualité des prédictions sur des données bruitées et variées comme les flux bancaires.

7. Analyse Approfondie : Évaluation (L’Heure de Vérité)
A. La Matrice de Confusion (Quadrants)
La matrice de confusion synthétise les performances de la manière suivante :

Vrais Positifs (TP) : transactions à risque correctement détectées comme à risque.

Vrais Négatifs (TN) : transactions normales correctement classées comme normales.

Faux Positifs (FP) : transactions normales classées par erreur comme à risque.

Faux Négatifs (FN) : transactions à risque classées par erreur comme normales.

Dans un système de détection de risque, les Faux Négatifs sont particulièrement critiques, car ils correspondent à des opérations problématiques non repérées.
Les Faux Positifs restent néanmoins importants à surveiller pour ne pas dégrader l’expérience client.

B. Les Métriques Avancées
L’accuracy seule peut être trompeuse lorsque la classe à risque est rare. Il est donc nécessaire de regarder :

Précision (Precision) : mesure la proportion de transactions réellement à risque parmi celles que le modèle a signalées. Une précision faible signifie trop de fausses alertes.

Rappel (Recall / Sensibilité) : mesure la proportion de transactions à risque correctement détectées parmi toutes les transactions à risque présentes dans les données. Un rappel faible signifie que le modèle laisse passer trop d’opérations dangereuses.

F1-Score : combine précision et rappel en une seule métrique. Il est particulièrement utile pour comparer des modèles lorsqu’il existe un déséquilibre de classes.

Conclusion du Projet
Ce projet montre que la Data Science appliquée aux transactions bancaires ne se limite pas à l’entraînement d’un modèle.
Il s’agit d’une chaîne cohérente de décisions : compréhension du contexte métier, préparation minutieuse des données, analyse exploratoire, définition d’un protocole expérimental robuste, choix d’un algorithme adapté (Random Forest) et interprétation rigoureuse des métriques d’évaluation.
