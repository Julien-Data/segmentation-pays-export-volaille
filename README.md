# Export Chicken Market Analysis

## 📌 Description du projet
Ce projet a pour objectif d’identifier des **groupes de pays pertinents pour l’exportation de poulets**, à partir de données issues de la FAO, de la Banque mondiale et d’autres sources open data.

L’approche combine :
- une **analyse PESTEL** pour guider le choix des variables,
- un **nettoyage et une fusion multi-sources**,
- une **Analyse en Composantes Principales (ACP)** pour simplifier la structure des données,
- un **clustering** (CAH et K-means) pour regrouper les pays selon leurs caractéristiques démographiques, agricoles et commerciales.

Le projet s’appuie sur deux notebooks :
1. **`1_preparation_donnees.ipynb`** – Préparation, nettoyage, normalisation
2. **`2_clustering_visualisations.ipynb`** – ACP, clustering, analyses finales

Les fonctions principales sont regroupées dans **`script.py`** et **`script2.py`**.

---

## 🗂 Dataset
- **Sources :**
  - FAO (Food and Agriculture Organization)
  - Banque mondiale
  - Données mondiales (Open Data)
- **Contenu :**
  - Indicateurs démographiques
  - Variables agricoles (production, disponibilité intérieure, importations…)
  - Indicateurs économiques et commerciaux
  - Variables enrichies grâce à l’analyse PESTEL

- **Traitements appliqués :**
  - Fusion et harmonisation des sources
  - Gestion des valeurs manquantes
  - Normalisation (scaling)
  - Sélection finale d’un jeu d’au moins **100 pays**, représentant plus de 60% de la population mondiale

> ⚠️ Données utilisées uniquement à des fins pédagogiques et analytiques.

---

## 🧪 Outils et bibliothèques utilisés
- **Python**
- **Jupyter Notebook**
- **Bibliothèques :**
  pandas, numpy, seaborn, matplotlib, scikit-learn, scipy, adjustText

- **Scripts Python :**
  - `script.py` : préparation/utilitaires
  - `script2.py` : ACP, clustering, visualisations avancées

---

## 🔍 Analyses réalisées

### 1. Préparation & nettoyage (`1_preparation_donnees.ipynb`)
- Uniformisation des sources FAO / Banque mondiale / Open Data
- Sélection des variables via PESTEL
- Normalisation des colonnes quantitatives
- Contrôle de cohérence & distributions

### 2. Analyse exploratoire
- Analyse statistique des variables
- Corrélations entre variables économiques et agricoles
- Exploration visuelle par pays

### 3. ACP (`2_clustering_visualisations.ipynb`)
- Analyse de la variance expliquée (ébouli)
- Cercle des corrélations
- Heatmap des loadings
- Projection des individus (pays)
- Visualisation combinée ACP + plan factoriel

### 4. Clustering
- **K-means :**
  - Méthode du coude (Elbow)
  - Silhouette score
  - Visualisation des centroides

- **CAH (Ward) :**
  - Dendrogramme multi-coupes
  - Clusters testés pour k = 2, 3, 4
  - Scores de silhouette pour sélectionner le meilleur modèle
  - Interprétation détaillée pour k = 4 (modèle retenu)

### 5. Analyse business
- Description des profils des clusters
- Identification des pays importateurs prioritaires
- Analyse des niveaux de production, disponibilité et demande
- Recommandations stratégiques pour l’export

---

## 📊 Résultats clés / Insights
- Les premières composantes de l’ACP capturent l’essentiel de la variabilité liée à la **production**, l’**offre intérieure** et les **importations**.
- Le clustering en **4 groupes** (CAH) propose une segmentation cohérente distinguant :
  - des **pays importateurs majeurs** (cibles prioritaires),
  - des pays **autosuffisants**,
  - des pays **en développement rapide**,
  - des pays à **faible potentiel commercial**.
- Le cluster des pays fortement dépendants des importations est le plus pertinent pour une stratégie d’exportation.

---

## 🚀 Objectifs atteints
- Construction d’une base multi-sources propre et normalisée
- Utilisation rigoureuse de l’**ACP** pour comprendre la structure des données
- Application d’un **clustering double (CAH + K-means)** pour une segmentation robuste
- Production de visualisations lisibles et exploitables
- Sélection d’un groupe de pays cible, justifié analytiquement

---