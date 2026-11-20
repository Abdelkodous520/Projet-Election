# Élections marocaines — données, analyses et tableau de bord

Mini-projet de **Programmation avancée pour l’IA** consacré aux élections communales et législatives au Maroc. Le dépôt réunit les données sources, un parcours pédagogique en notebooks, des analyses reproductibles et un tableau de bord interactif en français.

## Démarrage rapide

Python 3.10 ou une version plus récente est requis. Depuis la racine du projet :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS / Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
python Script/run_pipeline.py
python -m streamlit run app.py
```

Le tableau de bord s’ouvre dans le navigateur. La première commande régénère les fichiers nettoyés, tableaux, graphiques et rapports à partir des fichiers de `Data/`. Pour recalculer seulement les résultats communaux :

```bash
python Script/run_pipeline.py --communes-seulement
```

## Contenu du projet

```text
app.py                         Tableau de bord Streamlit
Data/                          Données sources et données communales nettoyées
Script/kharita_pipeline.py     Nettoyage et fonctions partagées
Script/run_pipeline.py        Point d’entrée pour régénérer les analyses
Script/collect_hespress_2026.py Collecte des résultats législatifs 2026 publiés par Hespress
Script/analyse_hespress_2026.py Analyse descriptive des exports 2026
Script/notebooks/             Parcours d’analyse communale en six étapes
Script/resultats/              Rapports et résultats législatifs
Script/resultats/communales/   Résultats communaux exportés
```

### Tableau de bord

Le tableau de bord présente deux analyses distinctes :

- **Communales (2015 et 2021)** : cartes interactives de la participation et de son évolution par commune, parts de voix et téléchargement du tableau comparatif.
- **Législatives (2011, 2016, 2021 et 2026)** : participation régionale et locale, parts de voix, partis arrivés en tête et profil des élus lorsque les données sont disponibles.

Les graphiques et indicateurs sont descriptifs. Les moyennes de participation ne sont pas pondérées par les inscrits. Les limites des communes proviennent des géométries incluses dans les données; le fond cartographique nécessite une connexion internet. Les classeurs législatifs ne fournissent pas de géométries adaptées à ces scrutins, leurs résultats restent donc comparés par région.

### Notebooks pédagogiques

Exécuter `python -m jupyter notebook` pour ouvrir Jupyter, puis parcourir les notebooks dans l’ordre :

1. `Script/notebooks/01_data_inspection_cleaning.ipynb` — inspection, anomalies et nettoyage.
2. `Script/notebooks/02_exploratory_data_analysis.ipynb` — participation et résultats des partis.
3. `Script/notebooks/03_electoral_evolution.ipynb` — comparaison des communes en 2015 et 2021.
4. `Script/notebooks/04_feature_engineering.ipynb` — parts de voix, concentration, marge et nombre de partis.
5. `Script/notebooks/05_machine_learning.ipynb` — régression et regroupement des profils.
6. `Script/notebooks/06_model_evaluation.ipynb` — métriques et interprétation des modèles.

Les classeurs législatifs complémentaires sont analysés par `Script/analyse_complete.py`. Le rapport correspondant est `Script/resultats/rapport_analyse.md`.

### Résultats législatifs 2026

Le collecteur crée deux CSV distincts : une ligne par circonscription et une ligne par liste de parti. Les résultats 2026 ne sont pas fusionnés avec les fichiers historiques, car la source, le découpage et l’état de publication peuvent différer.

```bash
python Script/collect_hespress_2026.py
python Script/analyse_hespress_2026.py
```

Les fichiers seront enregistrés dans `Data/hespress_2026/`, avec l’URL source et la date d’extraction dans `metadata.json`. Le portail peut refuser les requêtes automatisées; dans ce cas le collecteur s’arrête sans générer de résultat partiel.

## Données et provenance

- Le jeu principal, fourni dans `Data/resultats-elections-communales-maroc-2015-2021.csv`, provient de [Kharita — résultats des élections communales marocaines 2015–2021](https://lekharita.com/en/donnees/resultats-elections-communales-maroc-2015-2021). La source annonce une licence **CC BY 4.0**; l’attribution est conservée ici.
- Les classeurs législatifs 2011, 2016 et 2021 sont inclus dans `Data/` et analysés séparément des élections communales.
- Les résultats législatifs 2026 sont collectés depuis [Hespress Élections — circonscriptions 2026](https://elections.hespress.com/fr/elections/2026/circonscriptions), qui indique le ministère de l’Intérieur comme source. La collecte conserve l’URL et l’horodatage associés à chaque export.
- `Data/API_SG.GEN.PARL.ZS_DS2_en_csv_v2_4495.zip` est une source complémentaire World Bank; elle ne fait pas partie du parcours principal.

Les sources brutes ne sont pas modifiées. Le lecteur communal corrige le mélange d’encodages présent dans le CSV source afin de préserver à la fois les caractères accentués et les noms en arabe.

## Méthode et limites

- Les valeurs manquantes restent distinctes de zéro; une colonne de voix absente n’est pas interprétée comme un vote nul.
- La couverture des champs de voix est publiée avec les indicateurs calculés à partir des suffrages. L’indice de concentration et la marge ne sont établis que si les champs nécessaires sont complets.
- Le modèle supervisé explore la participation de 2021 à partir de caractéristiques de 2015. Avec seulement deux scrutins communaux, ses résultats ne constituent pas une prévision générale.
- Les scrutins législatifs sont agrégés à partir des listes locales pour les comparaisons de partis. Leurs découpages administratifs peuvent évoluer entre les années.
- Les associations et différences présentées sont descriptives et ne démontrent pas de causalité.

## Résultats générés

`python Script/run_pipeline.py` écrit les données communales nettoyées et le rapport qualité dans `Data/processed/`, les résultats communaux dans `Script/resultats/communales/`, puis régénère l’analyse législative dans `Script/resultats/`.
