"""Analyse descriptive harmonisee des elections legislatives marocaines.

Lit les trois classeurs de Data/ sans les modifier et produit les tableaux,
graphiques et un rapport Markdown dans Script/resultats/.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import FILES, load_year  # noqa: E402

OUT = Path(__file__).resolve().parent / "resultats"
META = {
    "idRegion", "idWilaya", "idPrefProv", "idSousPref", "idCirconscription",
    "region", "wilaya", "prefProv", "sousPref", "circonscription", "typeListe",
    "annee", "nSieges", "nInscrits", "txParticipation", "pctNuls", "invalide",
    "repPctFemmes", "repAge34", "repAge3544", "repAge4554", "repAge55",
    "repEduSans", "repEdu1aire", "repEdu2aire", "repEduSup",
}


def get_party_columns(frame: pd.DataFrame) -> list[str]:
    """Use each workbook's dictionary to identify vote-count columns."""
    cols = []
    for year, filename in FILES.items():
        if year not in set(frame["annee"]):
            continue
        dictionary = pd.read_excel(ROOT / "Data" / filename, sheet_name="dictionnaire")
        definitions = dictionary.set_index("Label")["Définition (FR)"].astype(str)
        cols.extend(label for label, definition in definitions.items()
                    if "Nombre de voix obtenu" in definition)
    return sorted(set(cols) - META)


def get_parties_by_year() -> dict[int, list[str]]:
    result = {}
    for year, filename in FILES.items():
        dictionary = pd.read_excel(ROOT / "Data" / filename, sheet_name="dictionnaire")
        definitions = dictionary.set_index("Label")["Définition (FR)"].astype(str)
        result[year] = [label for label, definition in definitions.items()
                        if "Nombre de voix obtenu" in definition and label not in META]
    return result


def pct(value, digits=1):
    return "n.d." if pd.isna(value) else f"{value:.{digits}%}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    frames = [load_year(year, filename) for year, filename in FILES.items()]
    data = pd.concat(frames, ignore_index=True, sort=False)
    parties = get_party_columns(data)
    parties_by_year = get_parties_by_year()
    numeric = ["txParticipation", "nSieges", "nInscrits", "pctNuls", "invalide",
               "repPctFemmes", "repAge34", "repAge3544", "repAge4554", "repAge55",
               "repEduSans", "repEdu1aire", "repEdu2aire", "repEduSup", *parties]
    for col in numeric:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    local = data[data["typeListe"].astype(str).str.lower().eq("locale")].copy()
    local = local.dropna(subset=["region", "txParticipation"])

    # Participation globale: taux moyen des circonscriptions locales (non pondere).
    turnout = local.groupby("annee").agg(
        taux_moyen=("txParticipation", "mean"),
        taux_median=("txParticipation", "median"),
        minimum=("txParticipation", "min"), maximum=("txParticipation", "max"),
        n_circonscriptions=("txParticipation", "count"),
    ).reset_index()
    turnout.to_csv(OUT / "participation_par_annee.csv", index=False, encoding="utf-8-sig")
    by_region = local.groupby(["annee", "region"], as_index=False).agg(
        taux_moyen=("txParticipation", "mean"), n_circonscriptions=("txParticipation", "count"))
    by_region.to_csv(OUT / "participation_par_region.csv", index=False, encoding="utf-8-sig")
    by_const = local[["annee", "region", "prefProv", "circonscription", "txParticipation"]].copy()
    by_const.to_csv(OUT / "participation_par_circonscription.csv", index=False, encoding="utf-8-sig")
    ranks = by_const.sort_values(["annee", "txParticipation"], ascending=[True, False])
    extremes = pd.concat([ranks.groupby("annee").head(5), ranks.groupby("annee").tail(5)])
    extremes.to_csv(OUT / "extremes_participation.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=turnout, x="annee", y="taux_moyen", color="#2878B5", ax=ax)
    ax.set(title="Participation moyenne par circonscription locale", xlabel="Scrutin", ylabel="Taux moyen (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout(); fig.savefig(OUT / "participation_par_annee.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=by_region, x="region", y="taux_moyen", hue="annee", ax=ax)
    ax.set(title="Participation moyenne par région et par scrutin", xlabel="Région", ylabel="Taux moyen (%)")
    ax.tick_params(axis="x", rotation=45); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout(); fig.savefig(OUT / "participation_par_region.png", dpi=160); plt.close(fig)

    # Votes des listes locales uniquement. Le taux de voix est calcule sur le
    # total des voix des partis declares dans les colonnes du classeur.
    national_rows, local_rows, winners, seat_rows = [], [], [], []
    for year, group in local.groupby("annee"):
        available = [c for c in parties_by_year[int(year)] if c in group.columns]
        votes = group[available].fillna(0).sum()
        total = votes.sum()
        for party, count in votes.items():
            national_rows.append({"annee": year, "parti": party, "voix": count,
                                  "part_voix": count / total if total else float("nan")})
        for (region, constituency), part in group.groupby(["region", "circonscription"], dropna=False):
            sums = part[available].fillna(0).sum()
            valid = sums[sums > 0]
            denominator = valid.sum()
            for party, count in valid.items():
                local_rows.append({"annee": year, "region": region, "circonscription": constituency,
                                   "parti": party, "voix": count,
                                   "part_voix": count / denominator if denominator else float("nan")})
            if len(valid):
                winner = valid.idxmax()
                winners.append({"annee": year, "region": region, "circonscription": constituency,
                                "parti_en_tete": winner, "voix_gagnant": valid[winner],
                                "part_voix_gagnant": valid[winner] / denominator})
    nat = pd.DataFrame(national_rows)
    loc = pd.DataFrame(local_rows)
    win = pd.DataFrame(winners)
    nat.to_csv(OUT / "resultats_partis_national_local.csv", index=False, encoding="utf-8-sig")
    loc.to_csv(OUT / "resultats_partis_circonscription.csv", index=False, encoding="utf-8-sig")
    win.to_csv(OUT / "parti_en_tete_circonscription.csv", index=False, encoding="utf-8-sig")
    if not nat.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        top = nat.sort_values("voix", ascending=False).groupby("annee").head(8)
        sns.barplot(data=top, x="parti", y="part_voix", hue="annee", ax=ax)
        ax.set(title="Parts des voix des partis — listes locales", xlabel="Parti", ylabel="Part des voix (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        fig.tight_layout(); fig.savefig(OUT / "parts_voix_partis.png", dpi=160); plt.close(fig)

    # Variation de voix et concentration HHI sur les listes locales.
    pivot = nat.pivot(index="parti", columns="annee", values="voix")
    common_years = [y for y in sorted(local.annee.unique())]
    comparable = pivot.dropna(subset=common_years) if len(common_years) else pivot
    changes = comparable.reset_index()
    if len(common_years) >= 2:
        changes["variation_voix_premier_dernier_scrutin"] = changes[common_years[-1]] - changes[common_years[0]]
        changes["variation_part_voix_pp"] = (
            nat.pivot(index="parti", columns="annee", values="part_voix")
            .loc[changes.parti, common_years[-1]].to_numpy()
            - nat.pivot(index="parti", columns="annee", values="part_voix")
            .loc[changes.parti, common_years[0]].to_numpy()) * 100
    changes.to_csv(OUT / "evolution_partis_presence_commune.csv", index=False, encoding="utf-8-sig")
    concentration = nat.groupby("annee").apply(lambda g: (g.part_voix ** 2).sum(), include_groups=False).rename("indice_HHI").reset_index()
    concentration.to_csv(OUT / "concentration_votes.csv", index=False, encoding="utf-8-sig")
    win_summary = win.groupby(["annee", "parti_en_tete"]).size().rename("circonscriptions_gagnees").reset_index()
    win_summary.to_csv(OUT / "victoires_par_parti.csv", index=False, encoding="utf-8-sig")
    if not win.empty:
        wide_winners = win.pivot(index=["region", "circonscription"], columns="annee", values="parti_en_tete")
        wide_winners.to_csv(OUT / "changements_parti_en_tete.csv", encoding="utf-8-sig")

    # Sommes des voix et sieges selon le type de liste (types restent separes).
    type_rows = []
    for (year, list_type), group in data.groupby(["annee", "typeListe"], dropna=False):
        available = [c for c in parties_by_year[int(year)] if c in group.columns]
        type_rows.append({"annee": year, "typeListe": list_type,
                          "nombre_lignes": len(group), "sieges_renseignes": group.nSieges.sum(min_count=1),
                          "voix_partis": group[available].fillna(0).to_numpy().sum()})
    types = pd.DataFrame(type_rows)
    types.to_csv(OUT / "resultats_par_type_liste.csv", index=False, encoding="utf-8-sig")

    # Votes nuls / invalides. Les deux classeurs ne nomment pas la variable
    # de la meme facon; aucun taux n'est calcule si le champ est absent.
    local["taux_nuls"] = local.get("pctNuls", pd.Series(index=local.index, dtype=float))
    if "invalide" in local:
        local["taux_nuls"] = local["taux_nuls"].fillna(local["invalide"])
    nulls = local.dropna(subset=["taux_nuls"])
    null_by_year = nulls.groupby("annee").agg(taux_nuls_moyen=("taux_nuls", "mean"),
                                                   n_circonscriptions=("taux_nuls", "count")).reset_index()
    null_by_year.to_csv(OUT / "votes_nuls_par_annee.csv", index=False, encoding="utf-8-sig")
    null_by_area = nulls[["annee", "region", "circonscription", "taux_nuls", "txParticipation"]]
    null_by_area.to_csv(OUT / "votes_nuls_par_zone.csv", index=False, encoding="utf-8-sig")
    def safe_corr(group):
        if group["taux_nuls"].nunique() < 2 or group["txParticipation"].nunique() < 2:
            return float("nan")
        return group["taux_nuls"].corr(group["txParticipation"])
    corr = nulls.groupby("annee").apply(safe_corr, include_groups=False)
    corr.rename("correlation_pearson_nuls_participation").to_csv(OUT / "correlation_nuls_participation.csv", encoding="utf-8-sig")

    # Profil des elus en 2021, moyenne des pourcentages a l'echelle des listes locales,
    # puis moyenne ponderee par sieges au niveau regional.
    profile_cols = ["repPctFemmes", "repAge34", "repAge3544", "repAge4554", "repAge55",
                    "repEduSans", "repEdu1aire", "repEdu2aire", "repEduSup"]
    prof = local[local.annee.eq(2021)].dropna(subset=["region"]).copy()
    prof_cols = [c for c in profile_cols if c in prof.columns]
    weighted_rows = []
    for region, group in prof.groupby("region"):
        row = {"region": region, "sieges": group.nSieges.sum(min_count=1)}
        for c in prof_cols:
            valid = group[[c, "nSieges"]].dropna()
            row[c] = (valid[c] * valid.nSieges).sum() / valid.nSieges.sum() if valid.nSieges.sum() else float("nan")
        weighted_rows.append(row)
    profile_region = pd.DataFrame(weighted_rows)
    profile_region.to_csv(OUT / "profil_elus_2021_par_region.csv", index=False, encoding="utf-8-sig")
    type_profile_rows = []
    for list_type, group in data[data.annee.eq(2021)].groupby("typeListe"):
        row = {"typeListe": list_type, "sieges": group.nSieges.sum(min_count=1)}
        for c in prof_cols:
            valid = group[[c, "nSieges"]].dropna()
            row[c] = (valid[c] * valid.nSieges).sum() / valid.nSieges.sum() if valid.nSieges.sum() else float("nan")
        type_profile_rows.append(row)
    pd.DataFrame(type_profile_rows).to_csv(OUT / "profil_elus_2021_par_type_liste.csv", index=False, encoding="utf-8-sig")
    prof[ ["region", "circonscription", "typeListe", "nSieges", *prof_cols] ].to_csv(
        OUT / "profil_elus_2021_detail.csv", index=False, encoding="utf-8-sig")
    if "repPctFemmes" in profile_region:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=profile_region, x="region", y="repPctFemmes", color="#D17A22", ax=ax)
        ax.set(title="Part des élues par région — listes locales, 2021", xlabel="Région", ylabel="Part des élues (%)")
        ax.tick_params(axis="x", rotation=45); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        fig.tight_layout(); fig.savefig(OUT / "part_elues_2021_region.png", dpi=160); plt.close(fig)

    # Donnees de controle et synthese lisible.
    participation_lines = []
    for row in turnout.itertuples():
        top = by_const[by_const.annee.eq(row.annee)].sort_values("txParticipation")
        lo, hi = top.iloc[0], top.iloc[-1]
        participation_lines.append(
            f"- **{row.annee}** : moyenne locale {pct(row.taux_moyen)}, médiane {pct(row.taux_median)} "
            f"(n={row.n_circonscriptions}); minimum {pct(lo.txParticipation)} à {lo.circonscription}, "
            f"maximum {pct(hi.txParticipation)} à {hi.circonscription}.")
    region_lines = []
    for year, group in by_region.groupby("annee"):
        low = group.loc[group.taux_moyen.idxmin()]
        high = group.loc[group.taux_moyen.idxmax()]
        region_lines.append(f"- **{year}** : région la plus élevée {high.region} ({pct(high.taux_moyen)}), "
                            f"la plus faible {low.region} ({pct(low.taux_moyen)}).")
    national_top = []
    for year, group in nat.groupby("annee"):
        top = group.sort_values("voix", ascending=False).head(5)
        national_top.append(f"- **{year}** : " + "; ".join(f"{r.parti} {r.voix:,.0f} voix ({pct(r.part_voix)})" for r in top.itertuples()) + ".")
    hhi_lines = [f"- **{r.annee}** : {r.indice_HHI:.3f}." for r in concentration.itertuples()]
    winner_lines = []
    for year, group in win_summary.groupby("annee"):
        top = group.sort_values("circonscriptions_gagnees", ascending=False).iloc[0]
        winner_lines.append(f"- **{year}** : {top.parti_en_tete} arrivé en tête dans {top.circonscriptions_gagnees} circonscriptions locales.")
    profile_dist = {}
    if not prof.empty:
        for col in ["repAge34", "repAge3544", "repAge4554", "repAge55", "repEduSans", "repEdu1aire", "repEdu2aire", "repEduSup"]:
            vals = prof[[col, "nSieges"]].dropna()
            profile_dist[col] = (vals[col] * vals.nSieges).sum() / vals.nSieges.sum()
    age_line = "; ".join(f"{label} {pct(profile_dist[col])}" for col, label in [
        ("repAge34", "≤34 ans"), ("repAge3544", "35–44 ans"), ("repAge4554", "45–54 ans"), ("repAge55", "≥55 ans")])
    edu_line = "; ".join(f"{label} {pct(profile_dist[col])}" for col, label in [
        ("repEduSans", "sans éducation formelle"), ("repEdu1aire", "primaire"),
        ("repEdu2aire", "secondaire"), ("repEduSup", "supérieur")])
    profile_summary = ""
    if not profile_region.empty:
        top_f = profile_region.sort_values("repPctFemmes", ascending=False).iloc[0]
        low_f = profile_region.sort_values("repPctFemmes").iloc[0]
        profile_summary = (f"En listes locales 2021, la moyenne pondérée par sièges de la part des élues est "
                           f"{pct((prof.repPctFemmes * prof.nSieges).sum() / prof.nSieges.sum())}. "
                           f"La région la plus élevée est {top_f.region} ({pct(top_f.repPctFemmes)}) et la plus basse "
                           f"{low_f.region} ({pct(low_f.repPctFemmes)}).")
    lines = [
        "# Analyse des élections législatives marocaines (2011, 2016, 2021)", "",
        "## Résultats principaux", "", "### Participation électorale", *participation_lines, "",
        "Moyennes régionales extrêmes :", *region_lines, "",
        "La moyenne est calculée sur les circonscriptions locales, sans pondération par les inscrits. "
        "Les classeurs ne fournissent pas `nInscrits` pour 2016 et 2021, donc ces valeurs ne sont pas des taux nationaux pondérés.", "",
        "### Résultats des partis", *national_top, "",
        "Parts de voix calculées comme les voix d'un parti divisées par la somme des voix déclarées pour les partis dans les listes locales. "
        "Les colonnes absentes selon les années ne sont pas assimilées à zéro pour la comparaison des partis présents dans tous les scrutins.", "",
        "### Évolution, leaders et concentration", "Les variations de voix des partis présents dans les trois scrutins figurent dans `evolution_partis_presence_commune.csv`. "
        "Le nombre de circonscriptions gagnées par parti est dans `victoires_par_parti.csv`; les changements de leader sont dans `changements_parti_en_tete.csv`.",
        *winner_lines, "Indice HHI des parts de voix locales (plus élevé = votes plus concentrés) :", *hhi_lines, "",
        "### Type de liste", "Les totaux sont ventilés par année et `typeListe` dans `resultats_par_type_liste.csv`. "
        "Les comparaisons directes entre types sont descriptives seulement : 2011 n'a que des lignes locales, 2016 comporte des lignes locales et nationales, "
        "et 2021 des lignes locales et régionales. Les sièges nationaux peuvent ne pas être renseignés.", "",
        "### Votes nuls ou invalides", "Les classeurs emploient des champs différents (`pctNuls` en 2011, `invalide` en 2021). "
        "Le fichier 2016 ne fournit pas de taux exploitable dans ce champ; toute comparaison temporelle complète serait donc trompeuse. "
        "En 2011, le taux moyen renseigné est " + pct(null_by_year.set_index("annee").loc[2011, "taux_nuls_moyen"]) +
        "; en 2021, les valeurs `invalide` observées sont nulles, ce qui rend la corrélation avec la participation indéfinie. "
        "Les données et corrélations sont dans les CSV `votes_nuls_*.csv` et `correlation_nuls_participation.csv`.", "",
        "### Profil des représentants en 2021", profile_summary, "Répartition pondérée par sièges : " + age_line + ". Niveau d'éducation : " + edu_line + ". "
        "Les indicateurs d'âge et d'éducation ainsi que les détails par circonscription sont dans `profil_elus_2021_*.csv`. "
        "Ils décrivent les élus et non les électeurs. Les proportions sont agrégées par région en les pondérant par le nombre de sièges locaux.", "",
        "## Limites et précautions", "- Les découpages et le nombre de circonscriptions peuvent différer; les noms de circonscriptions identiques ne garantissent pas toujours des frontières identiques.\n"
        "- Les résultats de partis sont en format large dans les classeurs; seuls les codes déclarés dans chaque dictionnaire sont utilisés.\n"
        "- Les résultats de voix agrégés aux niveaux national/local portent ici sur les listes locales pour garder une base comparable.\n"
        "- Les taux de votes nuls/invalides ne sont pas documentés de façon homogène dans les trois années.\n"
        "- Les représentations géographiques proposées sont des comparaisons par région; aucune carte n'est produite faute de géométrie géographique.", "",
        "## Fichiers produits", "Les tableaux CSV et graphiques PNG de cette analyse se trouvent dans le même dossier que ce rapport.", ""
    ]
    (OUT / "rapport_analyse.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Analyse terminée : {OUT}")
    print(f"Lignes locales : {len(local)}; colonnes de voix identifiées : {len(parties)}")


if __name__ == "__main__":
    main()
