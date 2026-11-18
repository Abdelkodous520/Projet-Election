# Analyse des élections législatives marocaines (2011, 2016, 2021)

## Résultats principaux

### Participation électorale
- **2011** : moyenne locale 48.7%, médiane 48.2% (n=92); minimum 32.5% à Al-Fida - Mers-Sultan, maximum 78.1% à Tarfaya.
- **2016** : moyenne locale 45.1%, médiane 44.2% (n=92); minimum 25.6% à Al-Fida - Mers-Sultan, maximum 76.7% à Aousserd.
- **2021** : moyenne locale 54.0%, médiane 58.1% (n=92); minimum 19.8% à Al-Fida - Mers-Sultan, maximum 85.1% à Tarfaya.

Moyennes régionales extrêmes :
- **2011** : région la plus élevée Laâyoune - Boujdour - Sakia-El-Hamra (66.4%), la plus faible Taza - Al Hoceima - Taounate (40.6%).
- **2016** : région la plus élevée Laâyoune - Sakia-El-Hamra (64.9%), la plus faible Casablanca - Settat (38.7%).
- **2021** : région la plus élevée Laâyoune - Sakia-El-Hamra (71.4%), la plus faible Casablanca - Settat (40.9%).

La moyenne est calculée sur les circonscriptions locales, sans pondération par les inscrits. Les classeurs ne fournissent pas `nInscrits` pour 2016 et 2021, donc ces valeurs ne sont pas des taux nationaux pondérés.

### Résultats des partis
- **2011** : PJD 1,080,913 voix (22.8%); PI 562,720 voix (11.9%); RNI 537,553 voix (11.3%); PAM 524,386 voix (11.1%); USFP 408,108 voix (8.6%).
- **2016** : PJD 1,571,659 voix (27.2%); PAM 1,205,444 voix (20.9%); PI 621,280 voix (10.8%); RNI 558,875 voix (9.7%); MP 409,085 voix (7.1%).
- **2021** : RNI 2,099,036 voix (27.7%); PAM 1,400,122 voix (18.5%); PI 1,278,420 voix (16.8%); USFP 598,293 voix (7.9%); MP 534,292 voix (7.0%).

Parts de voix calculées comme les voix d'un parti divisées par la somme des voix déclarées pour les partis dans les listes locales. Les colonnes absentes selon les années ne sont pas assimilées à zéro pour la comparaison des partis présents dans tous les scrutins.

### Évolution, leaders et concentration
Les variations de voix des partis présents dans les trois scrutins figurent dans `evolution_partis_presence_commune.csv`. Le nombre de circonscriptions gagnées par parti est dans `victoires_par_parti.csv`; les changements de leader sont dans `changements_parti_en_tete.csv`.
- **2011** : PJD arrivé en tête dans 41 circonscriptions locales.
- **2016** : PJD arrivé en tête dans 40 circonscriptions locales.
- **2021** : RNI arrivé en tête dans 59 circonscriptions locales.
Indice HHI des parts de voix locales (plus élevé = votes plus concentrés) :
- **2011** : 0.113.
- **2016** : 0.153.
- **2021** : 0.158.

### Type de liste
Les totaux sont ventilés par année et `typeListe` dans `resultats_par_type_liste.csv`. Les comparaisons directes entre types sont descriptives seulement : 2011 n'a que des lignes locales, 2016 comporte des lignes locales et nationales, et 2021 des lignes locales et régionales. Les sièges nationaux peuvent ne pas être renseignés.

### Votes nuls ou invalides
Les classeurs emploient des champs différents (`pctNuls` en 2011, `invalide` en 2021). Le fichier 2016 ne fournit pas de taux exploitable dans ce champ; toute comparaison temporelle complète serait donc trompeuse. En 2011, le taux moyen renseigné est 21.4%; en 2021, les valeurs `invalide` observées sont nulles, ce qui rend la corrélation avec la participation indéfinie. Les données et corrélations sont dans les CSV `votes_nuls_*.csv` et `correlation_nuls_participation.csv`.

### Profil des représentants en 2021
En listes locales 2021, la moyenne pondérée par sièges de la part des élues est 2.0%. La région la plus élevée est Marrakech - Safi (5.6%) et la plus basse Casablanca - Settat (0.0%).
Répartition pondérée par sièges : ≤34 ans 3.0%; 35–44 ans 15.1%; 45–54 ans 37.4%; ≥55 ans 44.6%. Niveau d'éducation : sans éducation formelle 1.0%; primaire 6.9%; secondaire 32.5%; supérieur 59.7%. Les indicateurs d'âge et d'éducation ainsi que les détails par circonscription sont dans `profil_elus_2021_*.csv`. Ils décrivent les élus et non les électeurs. Les proportions sont agrégées par région en les pondérant par le nombre de sièges locaux.

## Limites et précautions
- Les découpages et le nombre de circonscriptions peuvent différer; les noms de circonscriptions identiques ne garantissent pas toujours des frontières identiques.
- Les résultats de partis sont en format large dans les classeurs; seuls les codes déclarés dans chaque dictionnaire sont utilisés.
- Les résultats de voix agrégés aux niveaux national/local portent ici sur les listes locales pour garder une base comparable.
- Les taux de votes nuls/invalides ne sont pas documentés de façon homogène dans les trois années.
- Les représentations géographiques proposées sont des comparaisons par région; aucune carte n'est produite faute de géométrie géographique.

## Fichiers produits
Les tableaux CSV et graphiques PNG de cette analyse se trouvent dans le même dossier que ce rapport.
