"""Harmonise les classeurs électoraux présents dans ../Data et exporte un CSV."""

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUTPUT = Path(__file__).resolve().parent / "data" / "processed" / "elections_maroc.csv"
FILES = {
    2011: "parlement-elections-2011-1-0.xlsx",
    2016: "parlement-elections-2016-1-1.xlsx",
    2021: "parlement-elections-2021-1-0.xlsx",
}


def normalize_column(name: object) -> str:
    """Normalize common Excel accent/whitespace variations in column labels."""
    value = str(name).strip()
    replacements = {"é": "e", "è": "e", "ê": "e", "É": "E", "à": "a"}
    for source, target in replacements.items():
        value = value.replace(source, target)
    return re.sub(r"\s+", "_", value)


def load_year(year: int, filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Classeur introuvable : {path}")

    book = pd.ExcelFile(path)
    # Les noms d'onglet des classeurs 2011/2016 contiennent un caractère
    # accentué mal encodé dans certains environnements ("donn�es").
    # Le préfixe reste stable et identifie l'onglet de données.
    sheet = next((s for s in book.sheet_names if normalize_column(s).lower().startswith("donn")), None)
    if sheet is None:
        raise ValueError(f"Onglet de données introuvable dans {filename}: {book.sheet_names}")

    frame = pd.read_excel(path, sheet_name=sheet)
    frame.columns = [normalize_column(column) for column in frame.columns]
    frame["annee"] = year
    return frame


def main() -> None:
    combined = pd.concat(
        [load_year(year, filename) for year, filename in FILES.items()],
        ignore_index=True,
        sort=False,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"CSV créé : {OUTPUT}")
    print(f"Dimensions : {combined.shape[0]} lignes × {combined.shape[1]} colonnes")
    print("Années :", ", ".join(map(str, sorted(combined["annee"].unique()))))


if __name__ == "__main__":
    main()
