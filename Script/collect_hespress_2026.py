"""Collect published 2026 legislative results from Hespress into CSV files.

The source is a public results portal. The script records source URLs and the
UTC extraction time so the resulting CSV files remain auditable. It does not
modify historic source files already present in ``Data/``.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "hespress_2026"
INDEX_URL = "https://elections.hespress.com/fr/elections/2026/circonscriptions"
USER_AGENT = "ElectionAnalysisStudentProject/1.0 (public-data research)"


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def french_number(value: str | None) -> float | None:
    if not value:
        return None
    normalized = clean_text(value).replace("\u00a0", "").replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def request_page(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    if response.status_code in {401, 403} or "cloudflare" in response.text.lower():
        raise RuntimeError(
            "Le portail Hespress a refusé la requête automatisée. Réessayez plus tard "
            "ou utilisez un export autorisé par le site; ne contournez pas sa protection."
        )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def constituency_urls(index: BeautifulSoup) -> list[dict[str, str]]:
    result = []
    seen: set[str] = set()
    current_region = ""
    for element in index.find_all(["h2", "a"]):
        if element.name == "h2":
            current_region = clean_text(element.get_text(" ", strip=True)).split(" · ")[0]
            continue
        href = element.get("href", "")
        if "/elections/2026/circonscriptions/" not in href:
            continue
        url = urljoin(INDEX_URL, href)
        if url in seen:
            continue
        seen.add(url)
        result.append({
            "region": current_region,
            "circonscription": clean_text(element.get_text(" ", strip=True)).split(" sièges")[0],
            "url_source": url,
        })
    if not result:
        raise ValueError("Aucune circonscription n’a été détectée sur la page d’index.")
    if len(result) < 100:
        raise ValueError(f"Extraction incomplète : {len(result)} circonscriptions détectées au lieu du jeu attendu.")
    return result


def parse_summary(soup: BeautifulSoup, item: dict[str, str], extracted_at: str) -> dict:
    text = clean_text(soup.get_text(" ", strip=True))
    heading = soup.find("h1")
    name = clean_text(heading.get_text(" ", strip=True)) if heading else item["circonscription"]
    participation = re.search(r"Participation\s+([0-9.,\s]+)\s*%", text, re.IGNORECASE)
    seats = re.search(r"sièges\s+([0-9]+)", text, re.IGNORECASE)
    kind = re.search(r"\b(Locale|Régionale)\b", text)
    status = re.search(r"\b(Définitif[^|]*|Provisoire[^|]*)", text)
    return {
        "annee": 2026,
        "region": item["region"],
        "circonscription": name,
        "type_liste": kind.group(1) if kind else None,
        "statut": clean_text(status.group(1)) if status else None,
        "sieges_total": int(seats.group(1)) if seats else None,
        "participation_pct": french_number(participation.group(1)) if participation else None,
        "url_source": item["url_source"],
        "extrait_le_utc": extracted_at,
    }


def parse_lists(soup: BeautifulSoup, summary: dict) -> list[dict]:
    rows = []
    table = next(
        (
            candidate
            for candidate in soup.find_all("table")
            if "Parti" in candidate.get_text(" ", strip=True) and "Voix" in candidate.get_text(" ", strip=True)
        ),
        None,
    )
    if table is None:
        return rows
    for row in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if len(cells) < 5 or cells[0].lower() == "parti":
            continue
        rows.append({
            **summary,
            "parti": cells[0],
            "tete_de_liste": cells[1],
            "voix": french_number(cells[2]),
            "part_voix_pct": french_number(cells[3]),
            "sieges_obtenus": french_number(cells[4]),
        })
    return rows


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted_at = datetime.now(UTC).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"})
    items = constituency_urls(request_page(session, INDEX_URL))
    summaries, party_rows = [], []
    for position, item in enumerate(items, start=1):
        print(f"[{position}/{len(items)}] {item['circonscription']}")
        soup = request_page(session, item["url_source"])
        summary = parse_summary(soup, item, extracted_at)
        summaries.append(summary)
        party_rows.extend(parse_lists(soup, summary))
    return pd.DataFrame(summaries), pd.DataFrame(party_rows)


def save(summaries: pd.DataFrame, party_rows: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(OUTPUT / "circonscriptions_2026.csv", index=False, encoding="utf-8-sig")
    party_rows.to_csv(OUTPUT / "listes_2026.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source": INDEX_URL,
        "source_notice": "Hespress indique le ministère de l’Intérieur comme source des résultats.",
        "circonscriptions": len(summaries),
        "lignes_listes": len(party_rows),
        "extrait_le_utc": summaries.extrait_le_utc.iloc[0] if not summaries.empty else None,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraire les résultats législatifs 2026 publiés par Hespress.")
    parser.parse_args()
    summaries, party_rows = collect()
    save(summaries, party_rows)
    print(f"Export terminé : {len(summaries)} circonscriptions et {len(party_rows)} listes dans {OUTPUT}")


if __name__ == "__main__":
    main()
