import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from adjustText import adjust_text
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', None)


def charger_donnees():
    table_pays = pd.read_csv("table_pays.csv")
    X_scaled = pd.read_csv("X_scaled.csv")
    return table_pays, X_scaled

def tracer_eboulis(X_scaled):
    pca_temp = PCA()   
    pca_temp.fit(X_scaled)  
    
    explained_variance_ratio = pca_temp.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components = len(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_components + 1), cumulative_variance, marker='o')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance cumulée expliquée (%)")
    plt.title("Diagramme des éboulis (scree plot)")
    plt.grid(True)
    plt.axhline(y=80, color='r', linestyle='--', label="Seuil 80 %")
    plt.xticks(range(1, num_components + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Affichage des inerties par composante
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Composante {i+1} : {round(ratio, 2)} % de variance expliquée")


def analyser_valeurs_propres(pca, seuil_variance=0.8):
    """
    Analyse les valeurs propres d'une ACP pour choisir le nombre optimal de composantes.
    
    Affiche :
    - Les valeurs propres (variance expliquée par composante)
    - Le nombre de composantes avec valeur propre > 1 (critère de Kaiser)
    - Le nombre de composantes nécessaires pour atteindre un seuil de variance cumulée (ex : 80%)
    """
    valeurs_propres = pca.explained_variance_
    variance_expliquee = pca.explained_variance_ratio_
    variance_cumulee = np.cumsum(variance_expliquee)

    nb_kaiser = np.sum(valeurs_propres > 1)
    nb_seuil_variance = np.argmax(variance_cumulee >= seuil_variance) + 1

    print("\n🔍 Analyse des valeurs propres de l'ACP :")
    for i, (vp, ve, vc) in enumerate(zip(valeurs_propres, variance_expliquee, variance_cumulee), 1):
        print(f"  PC{i} : valeur propre = {vp:.2f}, variance expliquée = {ve*100:.2f}%, cumulée = {vc*100:.2f}%")

    print(f"\n✅ Critère de Kaiser : {nb_kaiser} composantes avec valeur propre > 1")
    print(f"✅ {nb_seuil_variance} composantes nécessaires pour expliquer {int(seuil_variance*100)}% de la variance")

    return nb_kaiser, nb_seuil_variance


def faire_acp(X_scaled):
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    print("  Variance expliquée par composante :", pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title("Projection des pays sur les 2 premières composantes principales")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.grid(True)
    plt.show()

    return X_pca, pca


def cercle_correlation(pca, X, comp_x=1, comp_y=2):
    features = X.columns
    comp_x_idx = comp_x - 1
    comp_y_idx = comp_y - 1

    correlations = pd.DataFrame(
        pca.components_[[comp_x_idx, comp_y_idx], :].T,
        columns=[f'PC{comp_x}', f'PC{comp_y}'],
        index=features
    )
    correlations['norm'] = np.sqrt(
        correlations[f'PC{comp_x}']**2 + correlations[f'PC{comp_y}']**2
    )

    correlations_filtered = correlations.sort_values(by='norm', ascending=False).head(12)
    features_filtered = correlations_filtered.index
    palette = sns.color_palette("hls", len(features_filtered))

    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)

    texts = []
    for i in range(len(features_filtered)):
        x = correlations_filtered[f'PC{comp_x}'].iloc[i]
        y = correlations_filtered[f'PC{comp_y}'].iloc[i]

        plt.arrow(0, 0, x, y,
                  head_width=0.02, head_length=0.02,
                  fc=palette[i], ec=palette[i])

        txt = plt.text(x, y, features_filtered[i],
                       color=palette[i], ha='center', va='center', fontsize=9)
        texts.append(txt)

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        force_text=1.8
    )

    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    plt.xlabel(f'PC{comp_x}')
    plt.ylabel(f'PC{comp_y}')
    plt.title(f'Cercle des corrélations (PC{comp_x} vs PC{comp_y}) - Top 12 variables')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()



def afficher_heatmap_loadings(pca, X, top_n=None):
    """
    Affiche une heatmap des coefficients (loadings) des variables sur les composantes principales.

    Paramètres :
    - pca : objet PCA déjà entraîné
    - X : DataFrame initial contenant les variables (non-scalées)
    - top_n : nombre de variables les plus importantes à afficher (optionnel)
    """
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"F{i+1}" for i in range(pca.n_components_)],
        index=X.columns
    )

    if top_n:
        # On sélectionne les top variables les plus "chargées" (norme la plus élevée)
        norms = (loadings**2).sum(axis=1).sort_values(ascending=False)
        loadings = loadings.loc[norms.head(top_n).index]

    plt.figure(figsize=(10, max(6, 0.4 * len(loadings))))
    sns.heatmap(loadings, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Loading"})
    plt.title("Contribution des variables aux composantes principales (loadings)")
    plt.xlabel("Composantes principales")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()



def afficher_projection_individus(X_pca, table_pays, comp_x=1, comp_y=2):
    """
    Affiche la projection des pays sur le plan factoriel (ACP), avec origine centrée.

    Paramètres :
        X_pca (np.ndarray) : Données projetées (ACP)
        table_pays (pd.DataFrame) : DataFrame contenant les noms des pays
        comp_x (int) : Numéro de la composante principale pour l'axe X
        comp_y (int) : Numéro de la composante principale pour l'axe Y
    """
    x_vals = X_pca[:, comp_x - 1]
    y_vals = X_pca[:, comp_y - 1]
    noms_pays = table_pays['Zone_x']

    # Définir les limites centrées
    x_max = max(abs(x_vals.min()), abs(x_vals.max()))
    y_max = max(abs(y_vals.min()), abs(y_vals.max()))

    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, s=80, color='skyblue', edgecolor='gray')

    # Affichage des noms des pays
    for i, pays in enumerate(noms_pays):
        plt.text(x_vals[i], y_vals[i], pays, fontsize=8, ha='center', va='center')

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.axvline(0, color='gray', linestyle='--', lw=1)

    # Centrage du plan
    plt.xlim(-x_max * 1.1, x_max * 1.1)
    plt.ylim(-y_max * 1.1, y_max * 1.1)

    plt.title(f"Projection des pays sur le plan factoriel (PC{comp_x} vs PC{comp_y})")
    plt.xlabel(f"Composante principale {comp_x}")
    plt.ylabel(f"Composante principale {comp_y}")
    plt.gca().set_aspect('equal')  # Pour un repère carré
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def tracer_elbow_kmeans(X_scaled, k_max=10):
    """
    Trace le graphique de l'inertie intraclasse en fonction du nombre de clusters K.
    Permet de déterminer le K optimal via la méthode du coude.

    Paramètres :
        X_scaled (np.ndarray): Données standardisées
        k_max (int): Nombre maximal de clusters à tester (par défaut 10)
    """
    inerties = []

    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inerties.append(kmeans.inertia_)

    # Tracé du graphe
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k_max + 1), inerties, marker='o')
    plt.xlabel("Nombre de clusters K")
    plt.ylabel("Inertie intra-classe")
    plt.title("Méthode du coude pour choisir K")
    plt.grid(True)
    plt.xticks(range(1, k_max + 1))
    plt.show()

    # Affichage des inerties pour info
    for k, inertia in enumerate(inerties, start=1):
        print(f"K = {k} --> Inertie = {round(inertia, 2)}")


def faire_clustering_kmeans(X_pca, table_pays, k=4, palette=None):
    kmeans = KMeans(n_clusters=k, random_state=42) 
    clusters = kmeans.fit_predict(X_pca)
    table_pays['Cluster_KMeans'] = clusters

    if palette is None:
        palette = sns.color_palette("Set2", k)  # Utiliser k pour la palette aussi

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette=palette, s=100)
    plt.title(f"Clustering des pays (K-means sur ACP) avec k={k}")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend(title='Cluster KMeans')
    plt.grid(True)
    plt.show()

    return table_pays, palette


def faire_cah(X_scaled, table_pays, ks=[2, 3, 4]):
    linkage_matrix = linkage(X_scaled, method='ward')

    for k in ks:
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        table_pays[f"Cluster_CAH_k{k}"] = clusters

    return table_pays


def dendrogram_multi_cut(X_scaled, ks=[2, 3, 4]):
    # Étape 1 : Calcul des liaisons
    linkage_matrix = linkage(X_scaled, method='ward')

    # Étape 2 : Tracer le dendrogramme
    plt.figure(figsize=(15, 6))
    dendrogram(linkage_matrix, color_threshold=0)  # pas de seuil de couleur par défaut

    # Étape 3 : Ajouter les lignes horizontales pour les différentes valeurs de k
    from scipy.cluster.hierarchy import fcluster

    distances = []
    for k in ks:
        # On récupère la hauteur de coupe correspondant à k clusters
        # en observant les distances de fusions
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        # Calcule le seuil de coupure correspondant à ce k
        # (attention : c'est une approximation simple ici)
        threshold = linkage_matrix[-(k - 1), 2]  # hauteur à laquelle fusionne le dernier groupe
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'k={k}')
    
    plt.title("Dendrogramme hiérarchique avec seuils multiples")
    plt.xlabel("Pays")
    plt.ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()




def evaluer_silhouette_cah(X_scaled, ks=[2, 3, 4], methode='ward'):
    """
    Calcule et affiche les Silhouette scores pour différents K en CAH.
    
    Paramètres :
        X_scaled (np.ndarray) : Données standardisées
        ks (list) : Liste des valeurs de k à tester
        methode (str) : Méthode de linkage pour la CAH (par défaut : 'ward')
    
    Retourne :
        dict : Dictionnaire {k: silhouette_score}
    """
    print("\n🔍 Évaluation du Silhouette Score pour la CAH :")
    
    linkage_matrix = linkage(X_scaled, method=methode)
    scores = {}

    for k in ks:
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  ➤ k = {k} → Silhouette Score = {score:.4f}")
    
    return scores




def decrire_clusters(table_pays, cluster_col='Cluster_KMeans'):
    """
    Affiche les caractéristiques moyennes des clusters.
    """
    colonnes_quantitatives = table_pays.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col not in colonnes_quantitatives:
        colonnes_quantitatives.append(cluster_col)

    description = table_pays.groupby(cluster_col)[colonnes_quantitatives].mean().round(2)
    print(f"\nDescription des clusters ({cluster_col}):\n")
    print(description)

def afficher_pays_par_cluster(table_pays, cluster_col='Cluster_KMeans'):
    """
    Affiche les pays regroupés dans chaque cluster.
    """
    print(f"\nPays par cluster ({cluster_col}):")
    clusters = table_pays.groupby(cluster_col)['Zone_x'].apply(list)
    for cluster_id, pays in clusters.items():
        print(f"\nCluster {cluster_id} ({len(pays)} pays) :")
        print(", ".join(sorted(pays)))


def visualiser_centroides_clusters(table_pays, cluster_col='Cluster_KMeans'):
    """
    Affiche un graphique en barres des moyennes des variables pour chaque cluster.
    """
    colonnes_num = table_pays.select_dtypes(include=np.number).columns.drop(cluster_col)
    moyennes = table_pays.groupby(cluster_col)[colonnes_num].mean()

    # On laisse pandas gérer la figure, pas besoin de plt.figure()
    ax = moyennes.T.plot(kind='bar', figsize=(14, 6))
    ax.set_title(f'Comparaison des moyennes des variables par cluster ({cluster_col})')
    ax.set_xlabel("Variables")
    ax.set_ylabel("Valeur moyenne")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title=cluster_col)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def extraire_top_pays_par_import(table_pays, cluster_id=2, cluster_col='Cluster_KMeans', top_n=10):
    """
    Extrait les top pays d'un cluster selon les importations.

    Paramètres :
        table_pays (pd.DataFrame) : Table des pays avec clustering
        cluster_id (int) : Identifiant du cluster à explorer
        cluster_col (str) : Colonne indiquant le clustering utilisé
        top_n (int) : Nombre de pays à retourner

    Retourne :
        pd.DataFrame : Top N pays du cluster, triés par 'Importations - Quantité'
    """
    # Sélection du cluster
    cluster = table_pays[table_pays[cluster_col] == cluster_id]

    # Vérification des colonnes nécessaires
    colonnes_necessaires = ['Zone_x', 'Importations - Quantité', 'Disponibilité intérieure', 'Production']
    for col in colonnes_necessaires:
        if col not in cluster.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans la table.")

    # Tri et sélection
    cluster_sorted = cluster.sort_values(by='Importations - Quantité', ascending=False)
    top_pays = cluster_sorted[colonnes_necessaires].head(top_n)

    return top_pays


def evaluer_silhouette_kmeans(X, k_min=2, k_max=10):
    """
    Calcule et affiche les Silhouette scores pour différents K (clustering KMeans).
    
    Paramètres :
        X (np.ndarray): Données (idéalement réduites par ACP)
        k_min (int): Nombre minimal de clusters à tester
        k_max (int): Nombre maximal de clusters à tester
    """
    scores = []

    print("\nÉvaluation du Silhouette score :")
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"K = {k} → Silhouette score = {round(score, 4)}")

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(range(k_min, k_max + 1), scores, marker='o')
    plt.xlabel("Nombre de clusters K")
    plt.ylabel("Silhouette score")
    plt.title("Score de Silhouette pour différents K")
    plt.grid(True)
    plt.xticks(range(k_min, k_max + 1))
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np
import pandas as pd

def afficher_acp_combinee(X_pca, pca, X, table_pays, comp_x=1, comp_y=2):
    """
    Affiche côte à côte :
    - La projection des pays sur le plan factoriel (sans noms)
    - Le cercle des corrélations
    """
    comp_x_idx = comp_x - 1
    comp_y_idx = comp_y - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ### ---  Partie 1 : Projection des pays (plan factoriel sans noms) ---
    x_vals = X_pca[:, comp_x_idx]
    y_vals = X_pca[:, comp_y_idx]

    ax1.axhline(0, color='grey', lw=1)
    ax1.axvline(0, color='grey', lw=1)
    ax1.scatter(x_vals, y_vals, s=60, color='skyblue', edgecolor='gray')

    ax1.set_title(f"Projection des pays (PC{comp_x} vs PC{comp_y})")
    ax1.set_xlabel(f"PC{comp_x}")
    ax1.set_ylabel(f"PC{comp_y}")
    ax1.grid(True)
    ax1.set_aspect('equal')

    ### ---  Partie 2 : Cercle des corrélations ---
    features = X.columns
    components = pca.components_[[comp_x_idx, comp_y_idx], :].T
    correlations = pd.DataFrame(components, columns=[f'PC{comp_x}', f'PC{comp_y}'], index=features)

    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    ax2.add_artist(circle)
    ax2.axhline(0, color='grey', lw=1)
    ax2.axvline(0, color='grey', lw=1)

    texts = []
    for i in range(correlations.shape[0]):
        x = correlations.iloc[i, 0]
        y = correlations.iloc[i, 1]
        ax2.arrow(0, 0, x, y, head_width=0.02, head_length=0.02, fc='red', ec='red')
        txt = ax2.text(x, y, correlations.index[i], fontsize=9, ha='center', va='center')
        texts.append(txt)

    adjust_text(
        texts,
        ax=ax2,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        force_text=1.8
    )

    ax2.set_title(f"Cercle des corrélations (PC{comp_x} vs PC{comp_y})")
    ax2.set_xlabel(f"PC{comp_x}")
    ax2.set_ylabel(f"PC{comp_y}")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid()
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()
