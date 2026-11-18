# Mini-projet — Analyse électorale marocaine

Projet du module **Programmation Avancée pour l’IA**. Le parcours principal utilise le dataset communal Kharita 2015–2021 fourni dans `Data/` pour réaliser six étapes : inspection et nettoyage, analyse exploratoire, évolution entre scrutins, création de variables, apprentissage automatique et évaluation.

## Données

- `Data/resultats-elections-communales-maroc-2015-2021.csv` : dataset principal Kharita.
- Les classeurs d’élections législatives restent présents pour une extension indépendante.
- Source : [Kharita — résultats des élections communales marocaines 2015–2021](https://lekharita.com/en/donnees/resultats-elections-communales-maroc-2015-2021). Licence annoncée : CC BY 4.0. Conserver l’attribution à la source.

Les fichiers bruts ne sont jamais modifiés. Le nettoyage conserve les valeurs manquantes; les champs absents ne sont pas assimilés à des votes nuls. Les parts de voix incluent une mesure de couverture. L’indice HHI est produit seulement lorsque les champs de vote de la ligne sont complets.

## Parcours des six tâches

1. `Script/notebooks/01_data_inspection_cleaning.ipynb` — types, manquants, doublons, anomalies et export nettoyé.
2. `Script/notebooks/02_exploratory_data_analysis.ipynb` — participation, régions, partis et corrélations.
3. `Script/notebooks/03_electoral_evolution.ipynb` — comparaison des mêmes communes en 2015 et 2021.
4. `Script/notebooks/04_feature_engineering.ipynb` — parts de voix, HHI, marge et nombre de partis.
5. `Script/notebooks/05_machine_learning.ipynb` — modèles de régression et clustering.
6. `Script/notebooks/06_model_evaluation.ipynb` — métriques, importances et profils des clusters.

`Script/kharita_pipeline.py` contient les fonctions partagées. La copie nettoyée et le rapport qualité sont enregistrés dans `Data/processed/`.

## Démarrage

Depuis la racine du projet :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook Script/notebooks/01_data_inspection_cleaning.ipynb
```

Exécuter ensuite les notebooks dans l’ordre. Le CSV brut Kharita est fourni dans `Data/`; aucune récupération réseau n’est nécessaire.

## Méthode et limites

La cible du modèle supervisé est la participation 2021, avec les variables de 2015 seulement. L’évaluation répartit les communes entre apprentissage et test. Le clustering utilise des profils observés sur les deux scrutins, car il s’agit d’une analyse non supervisée. Avec seulement deux élections, la régression constitue une expérimentation entre 2015 et 2021, pas une prévision générale des prochaines élections. Les associations statistiques ne démontrent pas une causalité.

Le dataset complémentaire [openAFRICA / TAFRA](https://open.africa/dataset/elections-communales-maroc-2021) peut servir à vérifier 2021, sans ajouter de doublons au jeu principal.
