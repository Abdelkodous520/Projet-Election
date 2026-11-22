"""
Dashboard Streamlit - Élections communales au Maroc (2015 vs 2021)
===================================================================

Emplacement conseillé :  Projet-Election/dashboard/app.py
Lancement            :  streamlit run dashboard/app.py
Dépendances          :  pip install streamlit plotly pandas numpy shapely

Le fichier cherche tout seul  Data/processed/kharita_communes_nettoye.csv
(et, si présents, models/metrics.csv, models/feature_importance.csv,
models/predictions.csv pour la section « Modèles ML »).
Si le CSV est introuvable, un bouton de chargement apparaît dans la barre latérale.

Conventions de design (identiques sur tous les graphiques)
- une couleur = une entité (un parti garde sa couleur partout, jamais selon son rang) ;
- années = rampe ordonnée (gris pour l'ancien scrutin, bleu foncé pour le récent) ;
- magnitude = rampe séquentielle bleue ; polarité = bleu / gris / rouge ;
- marques fines, grille en filet, pas de double axe, légende dès 2 séries.
"""
from __future__ import annotations

import io
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from shapely import wkt as _wkt
    from shapely.geometry import mapping as _mapping

    HAS_SHAPELY = True
except ImportError:  # la carte sera désactivée proprement
    HAS_SHAPELY = False

# ----------------------------------------------------------------------------
# 1. Design tokens
# ----------------------------------------------------------------------------
SURFACE = "#ffffff"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Palette catégorielle validée (ordre fixe, jamais cyclé)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b3b1a8"  # « Autres »
NEUTRAL_FILL = "#d9d8d0"

BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIV_BLUE, DIV_MID, DIV_RED = "#1c5cab", "#f0efec", "#cf3f3e"
GOOD_TXT, BAD_TXT = "#006300", "#d03b3b"

PCT_COLS = [
    "turnout_pct", "lead_share_pct", "women_elected_pct", "elected_age_ge55_pct",
    "elected_age_le34_pct", "elected_age_35_44_pct", "elected_age_45_54_pct", "elected_higher_edu_pct",
]


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


DIVERGING = [
    [0.0, DIV_BLUE], [0.25, _mix(DIV_BLUE, DIV_MID, 0.5)], [0.5, DIV_MID],
    [0.75, _mix(DIV_MID, DIV_RED, 0.5)], [1.0, DIV_RED],
]


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def text_on(fill: str) -> str:
    """Texte posé dans un aplat : noir sur jaune / teintes claires, blanc sinon."""
    return INK if fill.lower() in {"#eda100", "#86b6ef", "#b3b1a8", "#d9d8d0", "#cde2fb"} else "#ffffff"


def year_colors(years: list[int]) -> dict[int, str]:
    years = list(years)
    if len(years) == 1:
        return {years[0]: "#1c5cab"}
    if len(years) == 2:
        return {years[0]: "#8fb4e3", years[1]: "#184f95"}
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    idx = np.linspace(0, len(ramp) - 1, len(years)).round().astype(int)
    return {y: ramp[i] for y, i in zip(years, idx)}


def fr(x, nd: int = 1, suffix: str = "") -> str:
    """Format français : 1 234,5"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s + suffix


# ----------------------------------------------------------------------------
# 2. Page + CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Élections communales – Maroc",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, select, textarea {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: linear-gradient(180deg, #e9eef6 0, #f4f6fa 340px, #f4f6fa 100%); color: #0b0b0b; }
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton, .stAppDeployButton, [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e9ef; }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.brand { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.brand .logo { width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#0d366b,#256abf);
  display:flex; align-items:center; justify-content:center; font-size:18px; }
.brand .name { font-weight:700; font-size:.98rem; line-height:1.1; color:#0b0b0b; }
.brand .tag  { font-size:.72rem; color:#898781; }

.hero { background: linear-gradient(120deg, #0d366b 0%, #184f95 55%, #256abf 100%); border-radius:16px;
  padding:22px 28px 20px 28px; color:#fff; margin-bottom:18px; box-shadow:0 8px 24px rgba(13,54,107,.20); }
.hero .eyebrow { font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; opacity:.72; font-weight:600; }
.hero .t { font-size:1.9rem; font-weight:700; line-height:1.15; margin:.2rem 0 .1rem 0; letter-spacing:-0.02em; }
.hero .meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.chip { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.24); padding:3px 12px;
  border-radius:999px; font-size:.78rem; font-weight:500; }
.hero-title { font-size:1.6rem; font-weight:700; color:#0b0b0b; }

.kpi { background:#fff; border:1px solid #e6e9ef; border-top:3px solid #256abf; border-radius:12px;
  padding:14px 16px 12px 16px; min-height:132px; display:flex; flex-direction:column; justify-content:flex-start;
  box-shadow:0 1px 3px rgba(16,24,40,.05); }
.kpi .lab { font-size:.76rem; color:#52514e; font-weight:500; margin-bottom:8px; min-height:2.1em; }
.kpi .val { font-size:1.65rem; font-weight:700; line-height:1.1; color:#0b0b0b; white-space:nowrap; letter-spacing:-0.02em; }
.kpi .val small { font-size:.9rem; font-weight:600; color:#52514e; margin-left:4px; letter-spacing:0; }
.kpi .dlt { font-size:.78rem; margin-top:auto; padding-top:8px; font-weight:500; color:#52514e; }
.kpi .sub { font-size:.74rem; margin-top:auto; padding-top:8px; color:#898781; }

.chart-title { font-size:1rem; font-weight:650; color:#0b0b0b; margin:0 0 2px 0; letter-spacing:-0.01em; }
.chart-sub   { font-size:.82rem; color:#52514e; margin:0 0 8px 0; }
.chart-note  { font-size:.76rem; color:#898781; margin-top:2px; }

[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px; border-color:#e6e9ef;
  box-shadow:0 1px 3px rgba(16,24,40,.05); }
[data-testid="stSegmentedControl"] button, div[role="radiogroup"] label { font-weight:600; }
[data-testid="stSegmentedControl"] { margin-bottom:6px; }
.stDataFrame { border-radius:10px; }
/* couleur d'accent bleue, même si config.toml est absent */
[data-baseweb="tag"] { background-color:#184f95 !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color:#ffffff !important; }
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  background:#e7effa !important; border-color:#256abf !important; color:#0d366b !important; }
</style>
""",
    unsafe_allow_html=True,
)

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
_KEY = {"n": 0}


def plot(fig: go.Figure) -> None:
    _KEY["n"] += 1
    key = f"plt_{_KEY['n']}"
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key, theme=None)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key, theme=None)


def table(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {"height": height} if height else {}
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def card(fig: go.Figure, title: str, subtitle: str | None = None, note: str | None = None) -> None:
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()
    with box:
        html = f"<div class='chart-title'>{title}</div>"
        if subtitle:
            html += f"<div class='chart-sub'>{subtitle}</div>"
        st.markdown(html, unsafe_allow_html=True)
        plot(fig)
        if note:
            st.markdown(f"<div class='chart-note'>{note}</div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, legend: bool = True, xgrid: bool = False, ygrid: bool = True) -> go.Figure:
    fig.update_layout(
        font=dict(family=FONT, color=INK2, size=12),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin=dict(l=4, r=16, t=34 if legend else 12, b=4),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
    )
    fig.update_xaxes(showgrid=xgrid, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
                     ticks="", automargin=True, tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12))
    fig.update_yaxes(showgrid=ygrid, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)",
                     ticks="", automargin=True, tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12))
    return fig


def round_bars(fig: go.Figure, radius: int = 4) -> go.Figure:
    try:
        fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
    except Exception:
        pass
    return fig


# ----------------------------------------------------------------------------
# 3. Données
# ----------------------------------------------------------------------------
def candidate_bases() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here, here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]


def find_file(*parts: str) -> Path | None:
    for b in candidate_bases():
        p = b.joinpath(*parts)
        if p.exists():
            return p
    return None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in PCT_COLS:  # tout en pourcentage 0-100
        if c in df.columns and df[c].notna().any() and df[c].max() <= 1.5:
            df[c] = df[c] * 100
    df["year"] = df["year"].astype(int)
    df["commune_code"] = df["commune_code"].astype(str)
    for c in ["region_name", "province_name", "commune_type", "lead_party", "commune_name"]:
        if c in df.columns:
            df[c] = df[c].fillna("Non renseigné").astype(str)
    df["lead_key"] = df["lead_party"].str.strip().str.upper()
    df["margin_pts"] = df["winning_margin"] * 100
    return df


@st.cache_data(show_spinner="Chargement des données…")
def load_df(path_str: str | None, raw: bytes | None) -> pd.DataFrame:
    src = io.BytesIO(raw) if raw is not None else path_str
    return prepare(pd.read_csv(src))


def vote_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("votes_") and c != "votes_total"]


def vote_label(col: str) -> str:
    s = col[len("votes_"):]
    return "SAP" if s.startswith("sap") else s.upper()


def votes_by_label(d: pd.DataFrame) -> pd.DataFrame:
    """Voix par parti (lignes) et par année (colonnes)."""
    cols = vote_cols(d)
    tot = d.groupby("year")[cols].sum().T
    return tot.groupby(tot.index.map(vote_label)).sum()


def build_colors(df: pd.DataFrame) -> tuple[dict, dict]:
    """Couleur fixe par parti (selon le poids global dans les données, pas selon le filtre)."""
    v = votes_by_label(df)
    order = v.sum(axis=1).sort_values(ascending=False).index.tolist()
    c_votes = {lab: CATEGORICAL[i] for i, lab in enumerate(order[:8])}
    leads = df["lead_key"].value_counts()
    if sum(k in c_votes for k in leads.index[:5]) >= 3:
        c_lead = c_votes
    else:  # les noms de lead_party ne correspondent pas aux colonnes votes_* : classement par victoires
        c_lead = {k: CATEGORICAL[i] for i, k in enumerate(leads.index[:8])}
    return c_votes, c_lead


# ----------------------------------------------------------------------------
# 4. Chargement + filtres
# ----------------------------------------------------------------------------
path = find_file("Data", "processed", "kharita_communes_nettoye.csv")
raw_bytes = None
with st.sidebar:
    st.markdown("<div class='brand'><div class='logo'>🗳️</div><div><div class='name'>Élections Maroc</div>"
                "<div class='tag'>Communales 2015 · 2021</div></div></div>", unsafe_allow_html=True)
    st.markdown("##### Filtres")
    if path is None:
        up = st.file_uploader("Charger kharita_communes_nettoye.csv", type="csv")
        if up is None:
            st.markdown("<div class='hero-title'>Élections communales au Maroc</div>", unsafe_allow_html=True)
            st.info("Fichier introuvable. Place `kharita_communes_nettoye.csv` dans `Data/processed/` "
                    "ou charge-le dans la barre latérale.")
            st.stop()
        raw_bytes = up.getvalue()

df = load_df(str(path) if path else None, raw_bytes)
DATA_KEY = hashlib.md5(raw_bytes).hexdigest() if raw_bytes else f"{path}:{path.stat().st_mtime}"
COLOR_VOTES, COLOR_LEAD = build_colors(df)
YEARS_ALL = sorted(df["year"].unique().tolist())

with st.sidebar:
    sel_years = st.multiselect("Scrutin", YEARS_ALL, default=YEARS_ALL, help="Vide = rien à afficher")
    regions = sorted(df["region_name"].unique())
    sel_regions = st.multiselect("Région (vide = toutes)", regions)
    types = sorted(df["commune_type"].unique())
    sel_types = st.multiselect("Type de commune (vide = tous)", types)
    st.caption("Les cartes d'évolution comparent toujours le premier et le dernier scrutin.")

base = df.copy()  # filtré par région / type, pas par année (cartes d'évolution)
if sel_regions:
    base = base[base["region_name"].isin(sel_regions)]
if sel_types:
    base = base[base["commune_type"].isin(sel_types)]

if not sel_years:
    st.warning("Sélectionne au moins un scrutin dans la barre latérale.")
    st.stop()
d = base[base["year"].isin(sel_years)]
if d.empty:
    st.warning("Aucune commune ne correspond aux filtres.")
    st.stop()
sel_years = sorted(sel_years)
YC = year_colors(sel_years)

# ----------------------------------------------------------------------------
# 5. En-tête + KPI
# ----------------------------------------------------------------------------
chips = [
    "Scrutins " + " · ".join(map(str, sel_years)),
    f"{fr(d['commune_code'].nunique(), 0)} communes",
    (f"{len(sel_regions)} région(s) sélectionnée(s)" if sel_regions else "Toutes les régions"),
]
if sel_types:
    chips.append(" / ".join(sel_types))
chip_html = "".join("<span class='chip'>" + c + "</span>" for c in chips)
st.markdown(
    "<div class='hero'><div class='eyebrow'>Tableau de bord · Kharita</div>"
    "<div class='t'>Élections communales au Maroc</div>"
    "<div class='meta'>" + chip_html + "</div></div>",
    unsafe_allow_html=True,
)


def kpi(label: str, value: str, delta: float | None = None, ref: str = "", unit: str = " pts",
        up_good: bool | None = True, sub: str | None = None) -> str:
    mm = re.match(r"^(.*?)\s(%|pts)$", value)
    if mm:
        value = f"{mm.group(1)}<small>{mm.group(2)}</small>"
    dh = ""
    if delta is not None and not pd.isna(delta):
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        sign = "+" if delta > 0 else ("−" if delta < 0 else "")
        if up_good is None or delta == 0:
            color = INK2
        else:
            good = (delta > 0) == up_good
            color = GOOD_TXT if good else BAD_TXT
        dh = f"<div class='dlt' style='color:{color}'>{arrow} {sign}{fr(abs(delta))}{unit} {ref}</div>"
    sh = f"<div class='sub'>{sub}</div>" if sub else ""
    return f"<div class='kpi'><div class='lab'>{label}</div><div class='val'>{value}</div>{dh}{sh}</div>"


def yearly_mean(col: str) -> tuple[float | None, str]:
    if len(sel_years) < 2:
        return None, ""
    a, b = sel_years[0], sel_years[-1]
    ma, mb = d.loc[d.year == a, col].mean(), d.loc[d.year == b, col].mean()
    return mb - ma, f"vs {a}"


top_party = d["lead_party"].value_counts()
dp, rp = yearly_mean("turnout_pct")
dm, rm = yearly_mean("margin_pts")
dn, rn = yearly_mean("number_of_parties")

k = st.columns(5)
k[0].markdown(kpi("Communes", fr(d["commune_code"].nunique(), 0), sub=f"{fr(len(d), 0)} résultats"), unsafe_allow_html=True)
k[1].markdown(kpi("Participation moyenne", fr(d["turnout_pct"].mean(), 1, " %"), dp, rp), unsafe_allow_html=True)
k[2].markdown(kpi("Marge de victoire moyenne", fr(d["margin_pts"].mean(), 1, " pts"), dm, rm, up_good=None), unsafe_allow_html=True)
k[3].markdown(kpi("Partis par commune", fr(d["number_of_parties"].mean(), 1), dn, rn, unit="", up_good=None), unsafe_allow_html=True)
k[4].markdown(kpi("Parti le plus souvent en tête", str(top_party.index[0]),
                  sub=f"{fr(top_party.iloc[0] / len(d) * 100, 1, ' %')} des résultats"), unsafe_allow_html=True)
st.write("")

SECTIONS = ["Vue d'ensemble", "Partis", "Cartes", "Élus", "Modèles ML", "Données"]
try:
    section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], label_visibility="collapsed", key="section")
except AttributeError:  # anciennes versions de Streamlit
    section = st.radio("Section", SECTIONS, horizontal=True, label_visibility="collapsed")
section = section or SECTIONS[0]
st.write("")


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def fig_hist(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, y in enumerate(sel_years):
        s = d.loc[d.year == y, "turnout_pct"].dropna()
        fig.add_trace(go.Histogram(
            x=s, name=str(y), marker_color=YC[y], opacity=0.8 if len(sel_years) > 1 else 1,
            xbins=dict(start=0, end=100, size=2.5),
            hovertemplate=f"{y} · %{{x}} %<br>%{{y}} communes<extra></extra>"))
    for i, y in enumerate(sel_years):
        m = d.loc[d.year == y, "turnout_pct"].mean()
        fig.add_vline(x=m, line=dict(color=YC[y], width=2, dash="dash"))
        left = i % 2 == 0   # étiquettes de part et d'autre du trait, décalées en hauteur
        fig.add_annotation(x=m, xref="x", y=1.0 - 0.1 * (i // 2), yref="paper", showarrow=False,
                           text=f"Moy. {y} : {fr(m)} %", xanchor="right" if left else "left",
                           xshift=-6 if left else 6, font=dict(color=INK, size=11),
                           bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(barmode="overlay", bargap=0.04)
    fig.update_xaxes(title="Taux de participation (%)", ticksuffix=" %")
    fig.update_yaxes(title="Nombre de communes")
    return style(fig, 380, legend=len(sel_years) > 1)


def fig_dumbbell(d: pd.DataFrame) -> go.Figure:
    g = d.groupby(["region_name", "year"])["turnout_pct"].mean().unstack("year")
    g = g.sort_values(g.columns[-1])
    regs = g.index.tolist()
    yc = year_colors(list(g.columns))
    fig = go.Figure()
    if g.shape[1] > 1:
        xs, ys = [], []
        for r in regs:
            v = g.loc[r].dropna()
            if len(v) >= 2:
                xs += [v.min(), v.max(), None]
                ys += [r, r, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=AXIS, width=2),
                                 hoverinfo="skip", showlegend=False))
    for y in g.columns:
        fig.add_trace(go.Scatter(
            x=g[y], y=regs, mode="markers", name=str(y),
            marker=dict(size=11, color=yc[y], line=dict(color=SURFACE, width=2)),
            hovertemplate=f"<b>%{{y}}</b><br>{y} : %{{x:.1f}} %<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=regs, showgrid=False)
    fig.update_xaxes(title="Participation moyenne (%)", ticksuffix=" %", showgrid=True)
    return style(fig, max(380, 26 * len(regs) + 90), legend=g.shape[1] > 1, xgrid=True, ygrid=False)


def fig_scatter(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y in sel_years:
        s = d[d.year == y].dropna(subset=["turnout_pct", "margin_pts"])
        fig.add_trace(go.Scattergl(
            x=s["turnout_pct"], y=s["margin_pts"], mode="markers", name=str(y),
            marker=dict(size=7, color=YC[y], opacity=0.55, line=dict(color=SURFACE, width=0.5)),
            customdata=np.stack([s["commune_name"], s["region_name"], s["lead_party"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>En tête : %{customdata[2]}"
                          f"<br>Participation : %{{x:.1f}} %<br>Marge : %{{y:.1f}} pts<extra>{y}</extra>"))
    fig.update_xaxes(title="Taux de participation (%)", ticksuffix=" %", showgrid=True)
    fig.update_yaxes(title="Marge de victoire (points)")
    return style(fig, 400, legend=len(sel_years) > 1, xgrid=True)


CORR_LABELS = {
    "turnout_pct": "Participation", "margin_pts": "Marge de victoire", "party_concentration": "Concentration (HHI)",
    "number_of_parties": "Nombre de partis", "lead_share_pct": "Part du 1er", "seats_total": "Sièges",
    "observed_party_votes": "Voix exprimées",
}


def fig_corr(d: pd.DataFrame) -> go.Figure:
    cols = [c for c in CORR_LABELS if c in d.columns]
    c = d[cols].corr(method="spearman")
    names = [CORR_LABELS[x] for x in cols]
    fig = go.Figure(go.Heatmap(
        z=c.values, x=names, y=names, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        colorbar=dict(thickness=10, len=0.7, title="", outlinewidth=0, tickfont=dict(color=INK2)),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    for i, r in enumerate(cols):
        for j, cc in enumerate(cols):
            v = c.iloc[i, j]
            fig.add_annotation(x=names[j], y=names[i], text=f"{v:.2f}".replace(".", ","), showarrow=False,
                               font=dict(size=11, color="#ffffff" if abs(v) > 0.55 else INK))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return style(fig, 400, legend=False, ygrid=False)


def fig_size(d: pd.DataFrame) -> go.Figure | None:
    s = d.dropna(subset=["turnout_pct", "observed_party_votes"]).copy()
    if s["observed_party_votes"].nunique() < 5:
        return None
    s["q"] = pd.qcut(s["observed_party_votes"], 5, labels=False, duplicates="drop")
    n = int(s["q"].max()) + 1
    g = s.groupby(["q", "year"])["turnout_pct"].mean().unstack("year")
    cats = [f"{int(i) + 1}/{n}" for i in g.index]
    fig = go.Figure()
    for y in g.columns:
        fig.add_trace(go.Bar(x=cats, y=g[y], name=str(y), marker_color=YC.get(y, "#1c5cab"),
                             text=[fr(v) for v in g[y]], textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK2, size=11),
                             hovertemplate=f"{y} · taille %{{x}}<br>%{{y:.1f}} %<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_xaxes(title="Taille de la commune (quintile de voix exprimées, 1 = plus petites)")
    fig.update_yaxes(title="Participation moyenne (%)", range=[0, float(g.max().max()) * 1.18])
    return round_bars(style(fig, 380, legend=g.shape[1] > 1))


def fig_wins(d: pd.DataFrame) -> go.Figure:
    w = d.groupby(["lead_party", "year"]).size().unstack("year", fill_value=0)
    w = w.loc[w.sum(axis=1).sort_values(ascending=False).head(10).index]
    parties = w.index.tolist()
    fig = go.Figure()
    for y in w.columns:
        fig.add_trace(go.Bar(
            y=parties, x=w[y], orientation="h", name=str(y), marker_color=YC.get(y, "#1c5cab"),
            text=w[y], texttemplate="%{x:,}", textposition="outside", cliponaxis=False,
            textfont=dict(color=INK2, size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{y} : %{{x}} communes<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(categoryorder="array", categoryarray=parties, autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Communes remportées", showgrid=True, range=[0, float(w.max().max()) * 1.15])
    return round_bars(style(fig, max(340, 22 * len(parties) * max(1, len(w.columns)) + 90),
                            legend=len(w.columns) > 1, xgrid=True, ygrid=False))


def shares_table(d: pd.DataFrame) -> pd.DataFrame:
    v = votes_by_label(d)
    return v / v.sum() * 100


def fig_shares(d: pd.DataFrame) -> go.Figure:
    s = shares_table(d)
    top = s.max(axis=1).sort_values(ascending=False).head(8).index.tolist()
    fig = go.Figure()
    ys = [str(y) for y in s.columns]
    for lab in top:
        col = COLOR_VOTES.get(lab, OTHER)
        vals = s.loc[lab].values
        fig.add_trace(go.Bar(
            y=ys, x=vals, orientation="h", name=lab, marker=dict(color=col, line=dict(color=SURFACE, width=1.5)),
            text=[fr(v, 1, " %") if v >= 6 else "" for v in vals], textposition="inside", insidetextanchor="start",
            textfont=dict(color=text_on(col), size=12),
            hovertemplate=f"<b>{lab}</b> · %{{y}}<br>%{{x:.1f}} % des voix<extra></extra>", width=0.34))
    rest = 100 - s.loc[top].sum()
    fig.add_trace(go.Bar(
        y=ys, x=rest.values, orientation="h", name="Autres", marker=dict(color=OTHER, line=dict(color=SURFACE, width=1.5)),
        text=[fr(v, 1, " %") if v >= 6 else "" for v in rest.values], textposition="inside", insidetextanchor="start",
        textfont=dict(color=INK, size=12), width=0.34,
        hovertemplate="<b>Autres</b> · %{y}<br>%{x:.1f} % des voix<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.1)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", title="Part des voix exprimées", showgrid=False)
    return round_bars(style(fig, 70 + 95 * len(ys), legend=True, ygrid=False), 4)


def fig_share_change(d: pd.DataFrame) -> go.Figure | None:
    s = shares_table(d)
    if s.shape[1] < 2:
        return None
    a, b = s.columns[0], s.columns[-1]
    ch = (s[b] - s[a]).dropna()
    keep = s.max(axis=1).sort_values(ascending=False).head(12).index
    ch = ch.loc[ch.index.intersection(keep)].sort_values()
    fig = go.Figure()
    for name, mask, col in (("Hausse", ch > 0, DIV_BLUE), ("Baisse", ch <= 0, DIV_RED)):
        sub = ch[mask]
        fig.add_trace(go.Bar(
            y=sub.index, x=sub.values, orientation="h", name=name, marker_color=col,
            text=[("+" if v > 0 else "−") + fr(abs(v)) for v in sub.values], textposition="outside",
            cliponaxis=False, textfont=dict(color=INK2, size=11),
            hovertemplate="<b>%{y}</b><br>%{x:+.2f} points<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=ch.index.tolist(), showgrid=False)
    lim = float(ch.abs().max()) * 1.3 or 1
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title=f"Variation de la part des voix, {a} → {b} (points)", range=[-lim, lim], showgrid=True)
    return round_bars(style(fig, max(340, 28 * len(ch) + 100), legend=True, xgrid=True, ygrid=False))


def fig_margin_by_party(d: pd.DataFrame) -> go.Figure:
    top = d["lead_party"].value_counts().head(6).index.tolist()
    fig = go.Figure()
    for p in top:
        col = COLOR_LEAD.get(str(p).strip().upper(), OTHER)
        s = d.loc[d.lead_party == p, "margin_pts"].dropna()
        fig.add_trace(go.Box(
            y=s, name=str(p), marker_color=col, line=dict(color=col, width=1.5), fillcolor=rgba(col, 0.28),
            boxpoints=False, hovertemplate="<b>%{x}</b><br>%{y:.1f} pts<extra></extra>"))
    fig.update_yaxes(title="Marge de victoire (points)")
    fig.update_xaxes(showgrid=False)
    return style(fig, 380, legend=False)


def fig_concentration(d: pd.DataFrame) -> go.Figure:
    s = d.dropna(subset=["party_concentration", "number_of_parties"]).copy()
    bins = [0, 3, 5, 7, 9, 12, 1000]
    labs = ["1–3", "4–5", "6–7", "8–9", "10–12", "13 et +"]
    s["bin"] = pd.cut(s["number_of_parties"], bins=bins, labels=labs)
    fig = go.Figure()
    for lab in labs:
        v = s.loc[s["bin"] == lab, "party_concentration"]
        if len(v) == 0:
            continue
        fig.add_trace(go.Box(y=v, name=lab, marker_color="#256abf", line=dict(color="#256abf", width=1.5),
                             fillcolor=rgba("#256abf", 0.25), boxpoints=False,
                             hovertemplate="<b>%{x} partis</b><br>HHI %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="Nombre de partis en lice", showgrid=False)
    fig.update_yaxes(title="Concentration des votes (HHI)")
    return style(fig, 380, legend=False)


# ----------------------------------------------------------------------------
# 7. Cartes
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Préparation des polygones…")
def build_geojson(data_key: str, _df: pd.DataFrame, year: int, tol: float = 0.005) -> dict:
    sub = _df.loc[_df["year"] == year, ["commune_code", "geometry_wkt"]].dropna()
    feats = []
    for code, w in zip(sub["commune_code"], sub["geometry_wkt"]):
        try:
            g = _wkt.loads(w)
        except Exception:
            continue
        if g.is_empty:
            continue
        g = g.simplify(tol, preserve_topology=True)
        feats.append({"type": "Feature", "id": str(code), "properties": {}, "geometry": _mapping(g)})
    return {"type": "FeatureCollection", "features": feats}


def bbox_of(gj: dict, ids: set[str]) -> tuple[float, float, float, float] | None:
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for s in c:
                walk(s)

    for f in gj["features"]:
        if f["id"] in ids:
            walk(f["geometry"]["coordinates"])
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def view_for(gj: dict, ids: set[str]) -> tuple[dict, float]:
    bb = bbox_of(gj, ids)
    if not bb:
        return {"lat": 29.5, "lon": -8.0}, 4.3
    lon0, lat0, lon1, lat1 = bb
    span_lon, span_lat = max(lon1 - lon0, 0.05), max(lat1 - lat0, 0.05)
    z_lat = np.log2(650 / (1.42 * span_lat * 1.15))
    z_lon = np.log2(900 / (1.42 * span_lon * 1.15))
    return {"lat": (lat0 + lat1) / 2, "lon": (lon0 + lon1) / 2}, float(np.clip(min(z_lat, z_lon) - 0.1, 3.5, 9))


def draw_map(gj, data, color, label, *, scale=None, rng=None, cat_map=None, order=None, hover=None, height=640):
    data = data.copy()
    data["commune_code"] = data["commune_code"].astype(str)
    center, zoom = view_for(gj, set(data["commune_code"]))
    kw = dict(geojson=gj, locations="commune_code", color=color, hover_name="commune_name", hover_data=hover or {},
              center=center, zoom=zoom, opacity=0.92, labels={color: label})
    if cat_map is not None:
        kw.update(color_discrete_map=cat_map, category_orders={color: order or list(cat_map)})
    else:
        kw.update(color_continuous_scale=scale, range_color=rng)
    try:
        fig = px.choropleth_map(data, map_style="carto-positron", **kw)
    except AttributeError:
        fig = px.choropleth_mapbox(data, mapbox_style="carto-positron", **kw)
    fig.update_traces(marker_line_width=0.25, marker_line_color="rgba(255,255,255,0.7)")
    fig.update_layout(
        margin=dict(l=0, r=0, t=34 if cat_map is not None else 0, b=0), height=height,
        paper_bgcolor=SURFACE, font=dict(family=FONT, color=INK2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        coloraxis_colorbar=dict(title=dict(text=label, font=dict(size=11)), thickness=12, len=0.55,
                                outlinewidth=0, tickfont=dict(color=INK2, size=11)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
    )
    return fig


SEQ_INDICATORS = {
    "Taux de participation": ("turnout_pct", "Participation (%)", ":.1f"),
    "Marge de victoire": ("margin_pts", "Marge (points)", ":.1f"),
    "Concentration des votes (HHI)": ("party_concentration", "HHI", ":.2f"),
    "Part des voix du parti en tête": ("lead_share_pct", "Part du 1er (%)", ":.1f"),
}
MAP_CHOICES = list(SEQ_INDICATORS) + [
    "Parti en tête (3 principaux)", "Zones de force d'un parti",
    "Changement de parti en tête", "Évolution de la participation",
]


# ----------------------------------------------------------------------------
# 8. Sections
# ----------------------------------------------------------------------------
def page_overview() -> None:
    c1, c2 = st.columns(2)
    with c1:
        card(fig_hist(d), "Distribution de la participation",
             "Nombre de communes par tranche de 2,5 points ; les pointillés marquent la moyenne.")
    with c2:
        card(fig_dumbbell(d), "Participation moyenne par région",
             "Chaque point est une région ; le trait relie les scrutins.")
    c3, c4 = st.columns(2)
    with c3:
        card(fig_scatter(d), "Participation et marge de victoire",
             "Une commune = un point. Survole un point pour voir la commune et le parti en tête.")
    with c4:
        card(fig_corr(d), "Corrélations entre indicateurs",
             "Corrélation de rang (Spearman), robuste aux grandes villes qui écrasent les distributions.")
    fs = fig_size(d)
    if fs is not None:
        card(fs, "Participation selon la taille de la commune",
             "Communes réparties en 5 groupes de taille égale selon les voix exprimées.")


def page_parties() -> None:
    c1, c2 = st.columns(2)
    with c1:
        card(fig_wins(d), "Communes remportées par parti", "Les 10 partis les plus souvent en tête.")
    with c2:
        fc = fig_share_change(d)
        if fc is None:
            st.info("La variation des parts de voix demande au moins deux scrutins sélectionnés.")
        else:
            card(fc, "Gagnants et perdants en parts de voix", "Les 12 partis les plus importants, variation en points.")
    card(fig_shares(d), "Parts de voix des partis",
         "Les 8 premiers partis (couleur fixe par parti) et le reste regroupé dans « Autres ». "
         "Les listes sans appartenance (SAP) sont additionnées.")
    c3, c4 = st.columns(2)
    with c3:
        card(fig_margin_by_party(d), "Marge de victoire selon le parti en tête",
             "Boîte = 50 % des communes, trait central = médiane.")
    with c4:
        card(fig_concentration(d), "Concentration des votes selon le nombre de partis",
             "Plus le HHI est haut, plus les voix se concentrent sur peu de partis.")


def page_maps() -> None:
    if not HAS_SHAPELY:
        st.error("Installe shapely pour afficher les cartes :  pip install shapely")
        return
    with st.container():
        a, b, c = st.columns([2, 1, 1])
        choice = a.selectbox("Indicateur", MAP_CHOICES)
        needs_two = choice in ("Changement de parti en tête", "Évolution de la participation")
        year = b.selectbox("Scrutin", YEARS_ALL, index=len(YEARS_ALL) - 1, disabled=needs_two)
        compare = c.checkbox("Comparer côte à côte", value=False, disabled=needs_two or len(YEARS_ALL) < 2)

    ya, yb = YEARS_ALL[0], YEARS_ALL[-1]
    if needs_two and ya == yb:
        st.info("Un seul scrutin disponible : cette carte nécessite deux scrutins.")
        return

    if needs_two:
        gj = build_geojson(DATA_KEY, df, yb)
        A = base[base.year == ya].set_index("commune_code")
        B = base[base.year == yb].set_index("commune_code")
        common = A.index.intersection(B.index)
        if len(common) == 0:
            st.warning("Aucune commune commune aux deux scrutins avec les filtres actuels.")
            return
        if choice == "Changement de parti en tête":
            m = pd.DataFrame({
                "commune_code": common, "commune_name": B.loc[common, "commune_name"].values,
                "region_name": B.loc[common, "region_name"].values,
                "avant": A.loc[common, "lead_party"].values, "apres": B.loc[common, "lead_party"].values})
            m["Statut"] = np.where(m["avant"] == m["apres"], "Inchangé", "Changé")
            m["transition"] = m["avant"] + " → " + m["apres"]
            share = (m["Statut"] == "Changé").mean() * 100
            fig = draw_map(gj, m, "Statut", "Statut", cat_map={"Changé": CATEGORICAL[1], "Inchangé": NEUTRAL_FILL},
                           order=["Changé", "Inchangé"], hover={"region_name": True, "transition": True, "Statut": False})
            card(fig, f"Changement de parti en tête, {ya} → {yb}",
                 f"{fr(share)} % des {fr(len(m), 0)} communes comparables ont changé de parti en tête.",
                 "Seules les communes présentes aux deux scrutins sont comparées (clé : commune_code).")
            tr = m.loc[m["Statut"] == "Changé", "transition"].value_counts().head(10).sort_values()
            if len(tr):
                f2 = go.Figure(go.Bar(y=tr.index, x=tr.values, orientation="h", marker_color=CATEGORICAL[1],
                                      text=tr.values, textposition="outside", cliponaxis=False,
                                      textfont=dict(color=INK2, size=11),
                                      hovertemplate="<b>%{y}</b><br>%{x} communes<extra></extra>"))
                f2.update_xaxes(title="Nombre de communes", showgrid=True, range=[0, float(tr.max()) * 1.15])
                f2.update_yaxes(showgrid=False, categoryorder="array", categoryarray=tr.index.tolist())
                f2.update_layout(bargap=0.4)
                card(round_bars(style(f2, max(300, 30 * len(tr) + 90), legend=False, xgrid=True, ygrid=False)),
                     "Transitions les plus fréquentes", "Parti en tête au premier scrutin → parti en tête au dernier.")
        else:
            m = pd.DataFrame({
                "commune_code": common, "commune_name": B.loc[common, "commune_name"].values,
                "region_name": B.loc[common, "region_name"].values,
                "avant": A.loc[common, "turnout_pct"].values, "apres": B.loc[common, "turnout_pct"].values})
            m["delta"] = m["apres"] - m["avant"]
            m = m.dropna(subset=["delta"])
            lim = float(np.nanquantile(m["delta"].abs(), 0.95)) or 1.0
            fig = draw_map(gj, m, "delta", "Évolution (points)", scale=DIVERGING, rng=(-lim, lim),
                           hover={"region_name": True, "avant": ":.1f", "apres": ":.1f", "delta": ":+.1f"})
            card(fig, f"Évolution de la participation, {ya} → {yb}",
                 f"Bleu = hausse, rouge = baisse (points). Hausse dans {fr((m['delta'] > 0).mean() * 100)} % des communes ; "
                 f"échelle limitée à ±{fr(lim)} points (95e centile).")
        return

    years_to_draw = [ya, yb] if compare and ya != yb else [year]
    cols = st.columns(len(years_to_draw)) if len(years_to_draw) > 1 else [st.container()]

    def frame_for(y):
        return base[base.year == y]

    # paramètres communs (même échelle des deux côtés)
    allv = base[base.year.isin(years_to_draw)]
    extra = {}
    if choice in SEQ_INDICATORS:
        col, lab, fmt = SEQ_INDICATORS[choice]
        vals = allv[col].dropna()
        extra = dict(rng=(float(vals.quantile(0.02)), float(vals.quantile(0.98))) if len(vals) else None)
    elif choice == "Parti en tête (3 principaux)":
        top3 = allv["lead_party"].value_counts().head(3).index.tolist()
    else:
        opts = allv["lead_party"].value_counts().index.tolist()
        party = st.selectbox("Parti", opts)

    for cc, y in zip(cols, years_to_draw):
        with cc:
            sub = frame_for(y)
            if sub.empty:
                st.info(f"Aucune commune en {y} avec ces filtres.")
                continue
            gj = build_geojson(DATA_KEY, df, y)
            if choice in SEQ_INDICATORS:
                col, lab, fmt = SEQ_INDICATORS[choice]
                sub = sub.dropna(subset=[col])
                fig = draw_map(gj, sub, col, lab, scale=BLUE_SEQ, rng=extra["rng"],
                               hover={"region_name": True, "lead_party": True, col: fmt})
                card(fig, f"{choice} — {y}", "Échelle commune aux scrutins comparés (2e–98e centile)." if len(years_to_draw) > 1
                     else "Les valeurs extrêmes sont écrêtées (2e–98e centile) pour garder du contraste.")
            elif choice == "Parti en tête (3 principaux)":
                sub = sub.copy()
                sub["Parti"] = np.where(sub["lead_party"].isin(top3), sub["lead_party"], "Autres")
                cmap = {p: COLOR_LEAD.get(str(p).strip().upper(), OTHER) for p in top3}
                cmap["Autres"] = OTHER
                fig = draw_map(gj, sub, "Parti", "Parti en tête", cat_map=cmap, order=top3 + ["Autres"],
                               hover={"region_name": True, "turnout_pct": ":.1f", "Parti": False})
                card(fig, f"Parti en tête — {y}", "Trois couleurs au maximum : au-delà, les teintes deviennent "
                     "difficiles à distinguer sur une carte. Utilise « Zones de force d'un parti » pour les autres.")
            else:
                sub = sub.copy()
                sub["Parti"] = np.where(sub["lead_party"] == party, str(party), "Autres partis")
                colp = COLOR_LEAD.get(str(party).strip().upper(), CATEGORICAL[0])
                fig = draw_map(gj, sub, "Parti", "Zone", cat_map={str(party): colp, "Autres partis": NEUTRAL_FILL},
                               order=[str(party), "Autres partis"],
                               hover={"region_name": True, "lead_share_pct": ":.1f", "Parti": False})
                n_win = int((sub["lead_party"] == party).sum())
                card(fig, f"Où {party} arrive en tête — {y}",
                     f"{fr(n_win, 0)} communes sur {fr(len(sub), 0)} ({fr(n_win / len(sub) * 100)} %).")


def page_elus() -> None:
    e = d.dropna(subset=["women_elected_pct"])
    if e.empty:
        st.info("Le profil des élus n'est disponible que pour certains scrutins (2021). "
                "Sélectionne-le dans la barre latérale.")
        return
    e_all = df.dropna(subset=["women_elected_pct"])
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Femmes élues (moyenne des communes)", fr(e["women_elected_pct"].mean(), 1, " %")), unsafe_allow_html=True)
    k2.markdown(kpi("Élus avec enseignement supérieur", fr(e["elected_higher_edu_pct"].mean(), 1, " %")), unsafe_allow_html=True)
    k3.markdown(kpi("Élus de 34 ans et moins", fr(e["elected_age_le34_pct"].mean(), 1, " %")), unsafe_allow_html=True)
    st.write("")

    def region_bar(col: str, title_x: str) -> go.Figure:
        g = e.groupby("region_name")[col].mean().sort_values()
        nat = e_all[col].mean()
        fig = go.Figure(go.Bar(
            y=g.index, x=g.values, orientation="h", marker_color="#256abf",
            text=[fr(v) + " %" for v in g.values], textposition="outside", cliponaxis=False,
            textfont=dict(color=INK2, size=11), hovertemplate="<b>%{y}</b><br>%{x:.1f} %<extra></extra>"))
        fig.add_vline(x=nat, line=dict(color=INK2, width=1.5, dash="dash"),
                      annotation_text=f"Moyenne nationale : {fr(nat)} %", annotation_font=dict(color=INK2, size=11),
                      annotation_position="bottom right")
        fig.update_xaxes(title=title_x, ticksuffix=" %", showgrid=True, range=[0, float(g.max()) * 1.2])
        fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=g.index.tolist())
        fig.update_layout(bargap=0.4)
        return round_bars(style(fig, max(360, 28 * len(g) + 100), legend=False, xgrid=True, ygrid=False))

    c1, c2 = st.columns(2)
    with c1:
        card(region_bar("women_elected_pct", "Femmes parmi les élus (%)"), "Femmes élues par région",
             "Moyenne des communes de chaque région.")
    with c2:
        card(region_bar("elected_higher_edu_pct", "Élus avec enseignement supérieur (%)"),
             "Enseignement supérieur par région", "Moyenne des communes de chaque région.")

    cols_age = ["elected_age_le34_pct", "elected_age_35_44_pct", "elected_age_45_54_pct", "elected_age_ge55_pct"]
    names = ["34 ans et moins", "35–44 ans", "45–54 ans", "55 ans et plus"]
    g = e.groupby("region_name")[cols_age].mean()
    g = g.loc[g["elected_age_ge55_pct"].sort_values().index]
    fig = go.Figure()
    for col, nm, fill in zip(cols_age, names, ORDINAL4):
        vals = g[col].values
        fig.add_trace(go.Bar(
            y=g.index, x=vals, orientation="h", name=nm,
            marker=dict(color=fill, line=dict(color=SURFACE, width=1.5)),
            text=[fr(v, 0, " %") if v >= 9 else "" for v in vals], textposition="inside", insidetextanchor="start",
            textfont=dict(color=text_on(fill), size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{nm} : %{{x:.1f}} %<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", title="Part des élus", showgrid=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=g.index.tolist())
    card(round_bars(style(fig, max(380, 30 * len(g) + 110), legend=True, ygrid=False)),
         "Répartition par âge des élus, par région", "Rampe ordonnée : plus la teinte est foncée, plus les élus sont âgés.")


def page_ml() -> None:
    mdir = find_file("models")
    if mdir is None or not mdir.is_dir():
        st.info("Aucun dossier `models/` trouvé. Dès que le responsable ML a exporté ses résultats, dépose-y :\n\n"
                "- `metrics.csv` : une ligne par modèle (modèle, accuracy, F1, RMSE…)\n"
                "- `feature_importance.csv` : variable, importance\n"
                "- `predictions.csv` : commune_code, valeur réelle, prédiction")
        return
    m = mdir / "metrics.csv"
    if m.exists():
        mt = pd.read_csv(m)
        st.markdown("<div class='chart-title'>Comparaison des modèles</div>", unsafe_allow_html=True)
        table(mt)
        num = mt.select_dtypes("number").columns.tolist()
        name_col = next((c for c in mt.columns if c not in num), mt.columns[0])
        if num:
            metric = st.selectbox("Métrique à comparer", num)
            s = mt.sort_values(metric)
            fig = go.Figure(go.Bar(y=s[name_col], x=s[metric], orientation="h", marker_color="#256abf",
                                   text=[fr(v, 3) for v in s[metric]], textposition="outside", cliponaxis=False,
                                   textfont=dict(color=INK2, size=11)))
            fig.update_yaxes(showgrid=False)
            fig.update_xaxes(showgrid=True, title=metric)
            fig.update_layout(bargap=0.45)
            card(round_bars(style(fig, max(260, 40 * len(s) + 90), legend=False, xgrid=True, ygrid=False)), metric, "Par modèle.")
    fi = mdir / "feature_importance.csv"
    if fi.exists():
        t = pd.read_csv(fi)
        numc = t.select_dtypes("number").columns.tolist()
        if numc:
            namec = next((c for c in t.columns if c not in numc), t.columns[0])
            t = t.sort_values(numc[0]).tail(15)
            fig = go.Figure(go.Bar(y=t[namec], x=t[numc[0]], orientation="h", marker_color="#256abf",
                                   hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>"))
            fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=t[namec].tolist())
            fig.update_xaxes(showgrid=True, title="Importance")
            fig.update_layout(bargap=0.4)
            card(round_bars(style(fig, max(320, 28 * len(t) + 90), legend=False, xgrid=True, ygrid=False)),
                 "Importance des variables", "Les 15 variables les plus influentes.")
    pr = mdir / "predictions.csv"
    if pr.exists():
        p = pd.read_csv(pr)
        low = {c.lower(): c for c in p.columns}
        pred = next((low[c] for c in low if "pred" in c), None)
        true = next((low[c] for c in low if any(t in c for t in ("true", "reel", "réel", "actual", "y_test", "observ"))), None)
        if pred and true:
            fig = go.Figure()
            lo, hi = float(min(p[true].min(), p[pred].min())), float(max(p[true].max(), p[pred].max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prédiction parfaite",
                                     line=dict(color=AXIS, width=2, dash="dash"), hoverinfo="skip"))
            fig.add_trace(go.Scattergl(x=p[true], y=p[pred], mode="markers", name="Communes",
                                       marker=dict(size=7, color="#256abf", opacity=0.5),
                                       hovertemplate="Réel %{x:.2f}<br>Prédit %{y:.2f}<extra></extra>"))
            fig.update_xaxes(title="Valeur réelle", showgrid=True)
            fig.update_yaxes(title="Valeur prédite")
            card(style(fig, 420, legend=True, xgrid=True), "Prédictions contre valeurs réelles",
                 "Plus les points sont proches de la diagonale, meilleure est la prédiction.")
        else:
            st.info("`predictions.csv` : colonnes réelle / prédiction non reconnues.")
    if not (m.exists() or fi.exists() or pr.exists()):
        st.info("Le dossier `models/` existe mais ne contient pas encore `metrics.csv`, "
                "`feature_importance.csv` ou `predictions.csv`.")


def page_data() -> None:
    show = d.drop(columns=[c for c in ("geometry_wkt", "lead_key") if c in d.columns])
    st.markdown("<div class='chart-title'>Données filtrées</div>"
                f"<div class='chart-sub'>{fr(len(show), 0)} lignes · {show.shape[1]} colonnes</div>", unsafe_allow_html=True)
    table(show, height=460)
    st.download_button("Télécharger les données filtrées (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name="communes_filtrees.csv", mime="text/csv")
    with st.expander("Dictionnaire des colonnes"):
        dico = find_file("Data", "processed", "dictionnaire_colonnes.csv")
        if dico is not None:
            table(pd.read_csv(dico), height=420)
        else:
            q = pd.DataFrame({"colonne": df.columns, "type": df.dtypes.astype(str).values,
                              "% manquant": (df.isna().mean() * 100).round(1).values})
            table(q, height=420)
            st.caption("Génère `Data/processed/dictionnaire_colonnes.csv` avec le notebook pour avoir les descriptions.")


if section == "Vue d'ensemble":
    page_overview()
elif section == "Partis":
    page_parties()
elif section == "Cartes":
    page_maps()
elif section == "Élus":
    page_elus()
elif section == "Modèles ML":
    page_ml()
else:
    page_data()

st.markdown("<div class='chart-note' style='margin-top:1.5rem'>Source : Kharita – résultats des élections "
            "communales (2015, 2021). Traitement : Projet-Election.</div>", unsafe_allow_html=True)
