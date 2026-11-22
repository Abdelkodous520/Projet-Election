"""Outils partagés pour l'analyse des élections communales Kharita 2015–2021.

Les valeurs manquantes sont conservées; aucune cellule de vote n'est remplacée
par zéro. Les indicateurs calculés à partir des votes portent une couverture.
"""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "resultats-elections-communales-maroc-2015-2021.csv"
PROCESSED = ROOT / "Data" / "processed"


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value).strip())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return value


def load_raw(path=RAW):
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [normalize_name(c) for c in frame.columns]
    return frame


def clean_data(frame):
    """Normalize types and labels, report data quality, and preserve missingness."""
    df = frame.copy()
    numeric = [c for c in df if c.startswith("votes_")] + [
        "year", "seats_total", "turnout_pct", "votes_total", "lead_votes",
        "lead_share_pct", "women_elected_pct", "elected_age_ge55_pct",
        "elected_age_le34_pct", "elected_age_35_44_pct", "elected_age_45_54_pct",
        "elected_higher_edu_pct",
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("commune_code", "region_code", "province_code"):
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    for col in ("lead_party", "commune_name", "region_name", "province_name", "commune_type"):
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    if "lead_party" in df:
        df["lead_party"] = df["lead_party"].str.upper()

    source_columns = list(df.columns)
    votes = [c for c in df if c.startswith("votes_")]
    # Enlève l'identifiant technique votes_total des colonnes de partis.
    parties = [c for c in votes if c not in {"votes_total"}]
    df["vote_fields_reported"] = 0
    df["vote_fields_available"] = 0
    df["vote_coverage_pct"] = np.nan
    df["observed_party_votes"] = np.nan
    df["party_concentration"] = np.nan
    df["winning_margin"] = np.nan
    df["number_of_parties"] = np.nan
    # Le schéma inclut des colonnes de partis propres à chaque scrutin. Une
    # colonne entièrement vide pour une année est structurellement indisponible
    # cette année et ne doit pas invalider les indicateurs des communes.
    for year, idx in df.groupby("year").groups.items():
        active = [c for c in parties if df.loc[idx, c].notna().any()]
        if not active:
            continue
        values = df.loc[idx, active].clip(lower=0)
        complete = values.notna().all(axis=1)
        coverage = values.notna().sum(axis=1).div(len(active)).mul(100)
        total = values.sum(axis=1, min_count=1).where(values.sum(axis=1, min_count=1) > 0)
        shares = values.div(total, axis=0)
        df.loc[idx, "vote_fields_reported"] = values.notna().sum(axis=1)
        df.loc[idx, "vote_fields_available"] = len(active)
        df.loc[idx, "vote_coverage_pct"] = coverage
        df.loc[idx, "observed_party_votes"] = values.sum(axis=1, min_count=1)
        df.loc[idx, "party_concentration"] = shares.pow(2).sum(axis=1, min_count=len(active)).where(complete)
        top2 = shares.apply(lambda row: row.nlargest(2).values, axis=1)
        df.loc[idx, "winning_margin"] = top2.apply(
            lambda vals: float(vals[0] - vals[1]) if len(vals) > 1 else np.nan).where(complete)
        df.loc[idx, "number_of_parties"] = values.gt(0).sum(axis=1).where(complete)

    issues = {
        "rows": len(df), "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_commune_year": int(df.duplicated(["commune_code", "year"], keep=False).sum())
            if {"commune_code", "year"}.issubset(df.columns) else None,
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "turnout_out_of_range": int((~df["turnout_pct"].between(0, 100) & df["turnout_pct"].notna()).sum()),
        "negative_party_votes": int((df[parties] < 0).sum().sum()),
        "missing_values": frame.isna().sum().loc[lambda s: s.gt(0)].to_dict(),
    }
    return df, issues, parties


def make_comparison(df):
    """One row per commune with comparable 2015 and 2021 values."""
    cols = ["commune_code", "year", "turnout_pct", "lead_party", "region_name",
            "party_concentration", "winning_margin", "number_of_parties"]
    base = df[cols].drop_duplicates(["commune_code", "year"]).copy()
    wide = base.pivot(index="commune_code", columns="year")
    wide.columns = [f"{name}_{year}" for name, year in wide.columns]
    wide = wide.reset_index()
    if "turnout_pct_2015" in wide and "turnout_pct_2021" in wide:
        wide["variation_points"] = wide["turnout_pct_2021"] - wide["turnout_pct_2015"]
    if "lead_party_2015" in wide and "lead_party_2021" in wide:
        valid = wide["lead_party_2015"].notna() & wide["lead_party_2021"].notna()
        wide["party_changed"] = np.where(valid, wide["lead_party_2015"] != wide["lead_party_2021"], np.nan)
    return wide


def build_features(df, parties):
    """Feature table using 2015 predictors and 2021 participation target."""
    source = df[df["year"].eq(2015)].copy().set_index("commune_code")
    features = source[["turnout_pct", "region_name", "party_concentration", "winning_margin",
                       "number_of_parties", "vote_coverage_pct"]].rename(columns={
        "turnout_pct": "turnout_2015", "region_name": "region"})
    votes = source[parties].clip(lower=0)
    denominator = votes.sum(axis=1, min_count=1).where(votes.sum(axis=1, min_count=1) > 0)
    for col in parties:
        features[col.replace("votes_", "share_")] = votes[col] / denominator
    target = df[df["year"].eq(2021)].set_index("commune_code")["turnout_pct"].rename("turnout_2021")
    result = features.join(target, how="inner")
    return result.replace([np.inf, -np.inf], np.nan)


def save_clean_outputs(df, issues):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "kharita_communes_nettoye.csv", index=False, encoding="utf-8-sig")
    pd.Series(issues, name="value").to_json(PROCESSED / "rapport_qualite.json", force_ascii=False, indent=2)
    return PROCESSED
