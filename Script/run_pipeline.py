"""Rebuild all project outputs from the electoral source files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from kharita_pipeline import (  # noqa: E402
    clean_data,
    load_raw,
    make_comparison,
    save_clean_outputs,
)

COMMUNAL_OUTPUT = SCRIPT_DIR / "resultats" / "communales"


def _write_csv(frame: pd.DataFrame, name: str) -> None:
    COMMUNAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMMUNAL_OUTPUT / name, index=False, encoding="utf-8-sig")


def _communal_party_summary(frame: pd.DataFrame, parties: list[str]) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby("year", sort=True):
        active = [column for column in parties if group[column].notna().any()]
        if not active:
            continue
        totals = group[active].clip(lower=0).sum(min_count=1).dropna()
        denominator = totals.sum()
        for column, votes in totals.items():
            rows.append({
                "annee": int(year),
                "parti": column.removeprefix("votes_").upper(),
                "voix": float(votes),
                "part_voix": float(votes / denominator) if denominator else float("nan"),
            })
    return pd.DataFrame(rows)


def _write_communal_report(frame: pd.DataFrame, quality: dict, regional: pd.DataFrame) -> None:
    row_count = f"{len(frame):,}".replace(",", "\u202f")
    commune_count = f"{frame.commune_code.nunique():,}".replace(",", "\u202f")
    lines = [
        "# Résultats des élections communales marocaines (2015–2021)",
        "",
        "## Vue d’ensemble",
        "",
        f"Le jeu comprend **{row_count} lignes** pour "
        f"**{commune_count} communes** et les scrutins "
        f"{', '.join(map(str, quality['years']))}.",
        f"Le contrôle qualité a relevé {quality['duplicate_rows']} doublon(s) complet(s), "
        f"{quality['duplicate_commune_year']} doublon(s) commune-scrutin et "
        f"{quality['turnout_out_of_range']} taux de participation hors de l’intervalle 0–100.",
        "",
        "## Participation",
        "",
    ]
    annual = frame.groupby("year").turnout_pct.agg(["mean", "median", "count"])
    for year, values in annual.iterrows():
        rows = regional[regional.annee.eq(int(year))]
        if rows.empty:
            continue
        count = f"{int(values['count']):,}".replace(",", "\u202f")
        high = rows.loc[rows.taux_moyen.idxmax()]
        low = rows.loc[rows.taux_moyen.idxmin()]
        lines.append(
            f"- **{int(year)}** : moyenne communale de {values['mean']:.1f} % "
            f"(médiane {values['median']:.1f} %, n={count}); "
            f"moyenne régionale la plus élevée en {high.region_name} "
            f"({high.taux_moyen:.1f} %) et la plus faible en {low.region_name} "
            f"({low.taux_moyen:.1f} %)."
        )
    lines.extend([
        "",
        "Les moyennes sont calculées sur les communes ayant un taux renseigné et ne sont pas pondérées par le nombre d’inscrits.",
        "",
        "## Méthode et limites",
        "",
        "Les valeurs manquantes sont préservées. Les colonnes de votes absentes d’un scrutin ne sont pas considérées comme des votes nuls; les parts de voix sont rapportées aux voix déclarées dans les colonnes disponibles.",
        "Les indicateurs de concentration et de marge ne sont calculés que lorsque les champs de voix nécessaires sont complets. La comparaison 2015–2021 décrit les communes présentes aux deux scrutins; elle ne permet pas d’établir une causalité.",
        "",
        "## Fichiers",
        "",
        "Les tableaux et graphiques de cette analyse sont dans `Script/resultats/communales/`. Le rapport de qualité du nettoyage se trouve dans `Data/processed/rapport_qualite.json`.",
        "",
    ])
    report_path = ROOT / "Script" / "resultats" / "rapport_communales.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_communal_outputs() -> Path:
    raw = load_raw()
    frame, quality, parties = clean_data(raw)
    save_clean_outputs(frame, quality)

    turnout_year = frame.groupby("year", as_index=False).agg(
        taux_moyen=("turnout_pct", "mean"),
        taux_median=("turnout_pct", "median"),
        minimum=("turnout_pct", "min"),
        maximum=("turnout_pct", "max"),
        n_communes=("turnout_pct", "count"),
    )
    turnout_region = frame.dropna(subset=["region_name", "turnout_pct"]).groupby(
        ["year", "region_name"], as_index=False
    ).agg(taux_moyen=("turnout_pct", "mean"), n_communes=("turnout_pct", "count"))
    turnout_region = turnout_region.rename(columns={"year": "annee"})
    turnout_year = turnout_year.rename(columns={"year": "annee"})
    _write_csv(turnout_year, "participation_par_annee.csv")
    _write_csv(turnout_region, "participation_par_region.csv")

    comparison = make_comparison(frame)
    names = (frame.loc[frame.year.eq(2015), ["commune_code", "commune_name"]]
             .drop_duplicates("commune_code").set_index("commune_code").commune_name)
    comparison.insert(1, "commune_name", comparison.commune_code.map(names))
    _write_csv(comparison, "evolution_communes.csv")

    party_summary = _communal_party_summary(frame, parties)
    _write_csv(party_summary, "parts_voix_partis.csv")
    leading_parties = frame.dropna(subset=["lead_party"]).groupby(
        ["year", "lead_party"], as_index=False
    ).size().rename(columns={"year": "annee", "lead_party": "parti", "size": "communes_en_tete"})
    _write_csv(leading_parties, "partis_en_tete.csv")

    COMMUNAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(turnout_year.annee.astype(str), turnout_year.taux_moyen, color="#147D78", width=0.58)
    ax.set(title="Participation moyenne aux élections communales", xlabel="Scrutin", ylabel="Taux moyen (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMMUNAL_OUTPUT / "participation_par_annee.png", dpi=180)
    plt.close(fig)

    _write_communal_report(frame, quality, turnout_region)
    return COMMUNAL_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Régénérer les analyses électorales du projet.")
    parser.add_argument(
        "--communes-seulement",
        action="store_true",
        help="régénérer l’analyse communale sans relancer l’analyse législative",
    )
    args = parser.parse_args()

    communal_output = build_communal_outputs()
    print(f"Analyse communale terminée : {communal_output}")
    if not args.communes_seulement:
        from analyse_complete import main as build_legislative_outputs

        build_legislative_outputs()


if __name__ == "__main__":
    main()
