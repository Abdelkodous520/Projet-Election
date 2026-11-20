"""Build descriptive analysis outputs from the Hespress 2026 CSV exports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Data" / "hespress_2026"
OUTPUT = ROOT / "Script" / "resultats" / "legislatives_2026"


def main() -> None:
    constituencies_path = INPUT / "circonscriptions_2026.csv"
    lists_path = INPUT / "listes_2026.csv"
    if not constituencies_path.exists() or not lists_path.exists():
        raise FileNotFoundError(
            "Les exports 2026 sont absents. Exécutez d’abord : python Script/collect_hespress_2026.py"
        )

    constituencies = pd.read_csv(constituencies_path, encoding="utf-8-sig")
    lists = pd.read_csv(lists_path, encoding="utf-8-sig")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    local = constituencies[constituencies.type_liste.astype(str).str.casefold().eq("locale")].copy()

    annual = pd.DataFrame([{
        "annee": 2026,
        "taux_moyen": local.participation_pct.mean(),
        "taux_median": local.participation_pct.median(),
        "minimum": local.participation_pct.min(),
        "maximum": local.participation_pct.max(),
        "n_circonscriptions": local.circonscription.nunique(),
    }])
    annual.to_csv(OUTPUT / "participation_par_annee_2026.csv", index=False, encoding="utf-8-sig")

    by_region = local.groupby("region", as_index=False).agg(
        participation_moyenne_pct=("participation_pct", "mean"),
        circonscriptions=("circonscription", "nunique"),
        sieges=("sieges_total", "sum"),
    )
    by_region.to_csv(OUTPUT / "participation_par_region_2026.csv", index=False, encoding="utf-8-sig")

    national = lists.groupby("parti", as_index=False).agg(
        voix=("voix", "sum"),
        sieges=("sieges_obtenus", "sum"),
    )
    national["part_voix_pct"] = national.voix.div(national.voix.sum()).mul(100)
    national = national.sort_values(["sieges", "voix"], ascending=False)
    national.to_csv(OUTPUT / "resultats_partis_2026.csv", index=False, encoding="utf-8-sig")

    winners = (lists.sort_values(["circonscription", "voix"], ascending=[True, False])
               .drop_duplicates("circonscription")
               [["region", "circonscription", "type_liste", "parti", "voix", "part_voix_pct", "sieges_obtenus"]])
    winners.to_csv(OUTPUT / "partis_en_tete_2026.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Analyse des élections législatives marocaines 2026", "",
        "## Couverture", "",
        f"L’analyse porte sur {constituencies.circonscription.nunique()} circonscriptions et {len(lists)} listes publiées.",
        "Les résultats sont descriptifs et dépendent de l’état publié par la source à la date d’extraction indiquée dans `Data/hespress_2026/metadata.json`.",
        "", "## Fichiers produits", "",
        "- `participation_par_region_2026.csv` : participation moyenne et sièges par région.",
        "- `participation_par_annee_2026.csv` : participation globale sur les circonscriptions locales.",
        "- `resultats_partis_2026.csv` : voix et sièges agrégés par parti.",
        "- `partis_en_tete_2026.csv` : parti ayant obtenu le plus de voix dans chaque circonscription.",
    ]
    (OUTPUT / "rapport_2026.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Analyse 2026 terminée : {OUTPUT}")


if __name__ == "__main__":
    main()
