"""Interactive dashboard for Moroccan election results."""
from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
COMMUNAL = ROOT / "Script" / "resultats" / "communales"
LEGISLATIVE = ROOT / "Script" / "resultats"
INK = "#1F2933"
TEAL = "#147D78"
GOLD = "#D89B3C"

st.set_page_config(
    page_title="Élections marocaines | Tableau de bord",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: 'Aptos', 'Inter', 'Segoe UI', sans-serif; }
      .stApp { background: #f7f8f6; color: #1f2933; }
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }
      h1, h2, h3 { font-family: 'Aptos Display', 'Inter', 'Segoe UI', sans-serif; letter-spacing: -0.035em; }
      h1 { font-size: 2.45rem !important; font-weight: 800 !important; }
      [data-testid="stSidebar"] { background: #142f32; }
      [data-testid="stSidebar"] * { color: #f5f7f5; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid #e4e8e3; border-radius: 14px; padding: 16px 18px; }
      [data-testid="stMetricLabel"] { color: #66736f; }
      [data-testid="stMetricValue"] { color: #142f32; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #e4e8e3; border-radius: 14px; }
      .eyebrow { color: #147d78; font-size: .75rem; letter-spacing: .14em; font-weight: 700; text-transform: uppercase; }
      .subtle { color: #66736f; font-size: 1rem; }
      .source-note { background: #edf3ef; border-left: 3px solid #147d78; border-radius: 5px; padding: 12px 16px; color: #41524f; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Chargement des résultats…")
def _read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner="Préparation des limites régionales…")
def regional_geojson(geometries: pd.DataFrame) -> dict:
    features = []
    for region, group in geometries.groupby("region_name", sort=True):
        shapes = [wkt.loads(value) for value in group.geometry_wkt.dropna()]
        if not shapes:
            continue
        geometry = unary_union(shapes).simplify(0.002, preserve_topology=True)
        if geometry.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {"region_name": region},
            "geometry": mapping(geometry),
        })
    return {"type": "FeatureCollection", "features": features}


@st.cache_data(show_spinner=False)
def map_view(geometry_wkts: tuple[str, ...]) -> tuple[dict, float]:
    """Return a map centre and zoom level fitted to the selected regions."""
    bounds = [wkt.loads(geometry).bounds for geometry in geometry_wkts]
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    span = max(max_x - min_x, (max_y - min_y) * 1.35, 0.05)
    zoom = max(4.25, min(9.5, math.log2(360 / span) - 0.55))
    return {"lat": (min_y + max_y) / 2, "lon": (min_x + max_x) / 2}, zoom


def chart_style(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=32, b=12),
        font=dict(family="Aptos, Inter, Segoe UI, sans-serif", color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e4e8e3")
    fig.update_yaxes(gridcolor="#edf0ed", zeroline=False)
    return fig


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtle">{description}</p>', unsafe_allow_html=True)


def show_communal_dashboard() -> None:
    page_header(
        "Scrutins communaux · 2015 et 2021",
        "Le vote, région par région.",
        "Comparez simplement la participation, les indicateurs de voix et leur évolution entre les deux scrutins.",
    )
    communes = load_csv(ROOT / "Data" / "processed" / "kharita_communes_nettoye.csv")
    annual = load_csv(COMMUNAL / "participation_par_annee.csv")
    party_shares = load_csv(COMMUNAL / "parts_voix_partis.csv")
    evolution = load_csv(COMMUNAL / "evolution_communes.csv")

    available_years = sorted(communes.year.dropna().astype(int).unique())
    with st.sidebar:
        st.markdown("## Paramètres")
        years = st.multiselect("Scrutins", available_years, default=available_years, key="communal_years")
        area_names = ["Toutes les régions"] + sorted(communes.region_name.dropna().unique().tolist())
        area = st.selectbox("Région", area_names, key="communal_region")
    if not years:
        st.info("Choisissez au moins un scrutin pour afficher les résultats.")
        return

    selected = communes[communes.year.isin(years)].copy()
    if area != "Toutes les régions":
        selected = selected[selected.region_name.eq(area)]
    latest_year = max(years)
    mean_turnout = selected.turnout_pct.mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participation moyenne", f"{mean_turnout:.1f} %" if pd.notna(mean_turnout) else "—")
    c2.metric("Communes observées", f"{selected.commune_code.nunique():,}".replace(",", " "))
    c3.metric("Scrutins", " · ".join(str(year) for year in years))
    c4.metric("Régions couvertes", f"{selected.region_name.nunique():,}".replace(",", " "))

    map_indicators = {
        "Participation (%)": "turnout_pct",
        "Concentration des voix (HHI)": "party_concentration",
        "Couverture des votes (%)": "vote_coverage_pct",
    }
    map_indicator = st.selectbox("Indicateur à cartographier", list(map_indicators), key="communal_map_indicator")
    map_year_data = communes[communes.year.eq(latest_year)].copy()
    if area != "Toutes les régions":
        map_year_data = map_year_data[map_year_data.region_name.eq(area)].copy()
    metric_columns = list(map_indicators.values())
    map_year_data = map_year_data.groupby("region_name", as_index=False)[metric_columns].mean()
    map_shapes = communes[communes.year.eq(latest_year)]
    if area != "Toutes les régions":
        map_shapes = map_shapes[map_shapes.region_name.eq(area)]
    region_geometries = map_shapes[["region_name", "geometry_wkt"]].drop_duplicates()
    map_geojson = regional_geojson(region_geometries)
    current_map_center, current_map_zoom = map_view(tuple(region_geometries.geometry_wkt))
    map_left, map_right = st.columns(2, gap="large")
    with map_left:
        with st.container(border=True):
            st.subheader(f"{map_indicator} · {latest_year}")
            metric_column = map_indicators[map_indicator]
            metric_data = map_year_data.dropna(subset=[metric_column])
            fig = px.choropleth_map(
                metric_data,
                geojson=map_geojson,
                locations="region_name",
                featureidkey="properties.region_name",
                color=metric_column,
                hover_name="region_name",
                hover_data={
                    metric_column: ":.1f" if metric_column != "party_concentration" else ":.2f",
                },
                color_continuous_scale=["#f2eee6", "#e5b65b", TEAL],
                map_style="carto-positron",
                center=current_map_center,
                zoom=current_map_zoom,
                opacity=0.78,
                labels={metric_column: map_indicator},
            )
            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=0, b=0),
                font=dict(family="Aptos, Inter, Segoe UI, sans-serif", color=INK),
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(title=map_indicator, thickness=12),
            )
            if metric_column != "party_concentration":
                fig.update_layout(coloraxis_colorbar_ticksuffix=" %")
            st.plotly_chart(fig, use_container_width=True)
    with map_right:
        with st.container(border=True):
            st.subheader("Variation de participation · 2015–2021")
            comparison_detail = (
                communes[communes.year.eq(2015)]
                [["commune_code", "region_name", "geometry_wkt"]]
                .merge(
                    evolution[["commune_code", "turnout_pct_2015", "turnout_pct_2021", "variation_points"]],
                    on="commune_code",
                    how="inner",
                )
                .dropna(subset=["variation_points"])
            )
            if area != "Toutes les régions":
                comparison_detail = comparison_detail[comparison_detail.region_name.eq(area)].copy()
            comparison_geo = comparison_detail.groupby("region_name", as_index=False).agg(
                turnout_pct_2015=("turnout_pct_2015", "mean"),
                turnout_pct_2021=("turnout_pct_2021", "mean"),
                variation_points=("variation_points", "mean"),
            )
            comparison_shapes = regional_geojson(
                comparison_detail[["region_name", "geometry_wkt"]].drop_duplicates()
            )
            extent = max(float(comparison_geo.variation_points.abs().max()), 1.0)
            comparison_center, comparison_zoom = map_view(
                tuple(comparison_detail.geometry_wkt.dropna())
            )
            fig = px.choropleth_map(
                comparison_geo,
                geojson=comparison_shapes,
                locations="region_name",
                featureidkey="properties.region_name",
                color="variation_points",
                hover_name="region_name",
                hover_data={
                    "turnout_pct_2015": ":.1f",
                    "turnout_pct_2021": ":.1f",
                    "variation_points": ":+.1f",
                },
                color_continuous_scale=["#c65c54", "#f2eee6", TEAL],
                color_continuous_midpoint=0,
                range_color=[-extent, extent],
                map_style="carto-positron",
                center=comparison_center,
                zoom=comparison_zoom,
                opacity=0.78,
                labels={
                    "variation_points": "Variation (points)",
                    "turnout_pct_2015": "Participation 2015 (%)",
                    "turnout_pct_2021": "Participation 2021 (%)",
                },
            )
            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=0, b=0),
                font=dict(family="Aptos, Inter, Segoe UI, sans-serif", color=INK),
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(title="Points", thickness=12),
            )
            st.plotly_chart(fig, use_container_width=True)
    map_scope = "l’ensemble du Maroc" if area == "Toutes les régions" else area
    st.caption(
        f"Cartes centrées sur {map_scope}. Chaque contour représente une région; "
        "les valeurs sont des moyennes régionales. Le rouge indique une baisse de participation "
        "et le vert une hausse."
    )

    trend = annual[annual.annee.isin(years)].copy()
    if area != "Toutes les régions":
        local_trend = selected.groupby("year", as_index=False).turnout_pct.mean().rename(
            columns={"year": "annee", "turnout_pct": "taux_moyen"}
        )
        trend = local_trend
    trend["annee"] = trend.annee.astype(str)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Participation au fil des scrutins")
            fig = px.line(
                trend,
                x="annee",
                y="taux_moyen",
                markers=True,
                labels={"annee": "Scrutin", "taux_moyen": "Participation moyenne (%)"},
                color_discrete_sequence=[TEAL],
            )
            fig.update_traces(line_width=3, marker_size=10)
            fig.update_yaxes(range=[0, 100], ticksuffix=" %")
            st.plotly_chart(chart_style(fig, 335), use_container_width=True)
    with right:
        with st.container(border=True):
            st.subheader(f"Participation par région · {latest_year}")
            regional = (
                communes[communes.year.eq(latest_year)]
                .groupby("region_name", as_index=False)
                .turnout_pct.mean()
                .sort_values("turnout_pct")
            )
            fig = px.bar(
                regional,
                x="turnout_pct",
                y="region_name",
                orientation="h",
                labels={"turnout_pct": "Participation moyenne (%)", "region_name": "Région"},
                color="turnout_pct",
                color_continuous_scale=["#d6e8e3", TEAL],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(range=[0, 100], ticksuffix=" %")
            st.plotly_chart(chart_style(fig, 335), use_container_width=True)

    st.subheader("Parts de voix par parti")
    selected_parties = party_shares[party_shares.annee.isin(years)].copy()
    if not selected_parties.empty:
        top_parties = (selected_parties.groupby("parti").voix.sum().nlargest(10).index)
        selected_parties = selected_parties[selected_parties.parti.isin(top_parties)]
        selected_parties["annee"] = selected_parties.annee.astype(str)
        fig = px.bar(
            selected_parties,
            x="parti",
            y="part_voix",
            color="annee",
            barmode="group",
            labels={"parti": "Parti", "part_voix": "Part des voix", "annee": "Scrutin"},
            color_discrete_sequence=[TEAL, GOLD],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig, 360), use_container_width=True)
    else:
        st.info("Aucun résultat de parti disponible pour cette sélection.")

    st.subheader("Ce qui a changé entre 2015 et 2021")
    comparable = evolution.dropna(subset=["variation_points"]).copy()
    if area != "Toutes les régions":
        codes = set(communes.loc[communes.region_name.eq(area), "commune_code"])
        comparable = comparable[comparable.commune_code.isin(codes)]
    gains, losses = st.columns(2)
    with gains:
        st.caption("Plus fortes hausses de participation")
        st.dataframe(
            comparable.nlargest(5, "variation_points")[["commune_name", "region_name_2015", "variation_points"]]
            .rename(columns={"commune_name": "Commune", "region_name_2015": "Région", "variation_points": "Variation (points)"}),
            hide_index=True,
            use_container_width=True,
        )
    with losses:
        st.caption("Plus fortes baisses de participation")
        st.dataframe(
            comparable.nsmallest(5, "variation_points")[["commune_name", "region_name_2015", "variation_points"]]
            .rename(columns={"commune_name": "Commune", "region_name_2015": "Région", "variation_points": "Variation (points)"}),
            hide_index=True,
            use_container_width=True,
        )
    st.download_button(
        "Télécharger la comparaison des communes",
        comparable.to_csv(index=False).encode("utf-8-sig"),
        file_name="evolution_communes_2015_2021.csv",
        mime="text/csv",
    )
    st.markdown(
        '<div class="source-note">Source : Kharita, résultats des élections communales marocaines 2015–2021. '
        'Les moyennes ne sont pas pondérées par les inscrits. Les valeurs de vote manquantes restent manquantes.</div>',
        unsafe_allow_html=True,
    )


def show_legislative_dashboard() -> None:
    page_header(
        "Scrutins législatifs · 2011, 2016, 2021 et 2026",
        "Trois élections, un pays en mouvement.",
        "Parcourez la participation locale, les partis arrivés en tête et les profils des élus.",
    )
    annual = load_csv(LEGISLATIVE / "participation_par_annee.csv")
    regions = load_csv(LEGISLATIVE / "participation_par_region.csv")
    parties = load_csv(LEGISLATIVE / "resultats_partis_national_local.csv")
    wins = load_csv(LEGISLATIVE / "victoires_par_parti.csv")
    profile = load_csv(LEGISLATIVE / "profil_elus_2021_par_region.csv")

    output_2026 = LEGISLATIVE / "legislatives_2026"
    has_2026 = (output_2026 / "participation_par_annee_2026.csv").exists()
    if has_2026:
        annual_2026 = load_csv(output_2026 / "participation_par_annee_2026.csv")
        regions_2026 = load_csv(output_2026 / "participation_par_region_2026.csv")
        parties_2026 = load_csv(output_2026 / "resultats_partis_2026.csv")
        wins_2026 = load_csv(output_2026 / "partis_en_tete_2026.csv")
        # Les exports Hespress expriment les taux en points (41,7 = 41,7 %),
        # alors que les historiques du projet utilisent des fractions (0,417).
        for column in ["taux_moyen", "taux_median", "minimum", "maximum"]:
            annual_2026[column] = annual_2026[column].div(100)
        regions_2026 = regions_2026.rename(columns={"participation_moyenne_pct": "taux_moyen"})
        regions_2026["taux_moyen"] = regions_2026.taux_moyen.div(100)
        regions_2026["annee"] = 2026
        parties_2026 = parties_2026.rename(columns={"part_voix_pct": "part_voix"})
        parties_2026["part_voix"] = parties_2026.part_voix.div(100)
        parties_2026["annee"] = 2026
        wins_2026 = wins_2026.rename(columns={"parti": "parti_en_tete"})
        wins_2026["annee"] = 2026
        wins_2026["circonscriptions_gagnees"] = 1
        wins_2026 = wins_2026.groupby(["annee", "parti_en_tete"], as_index=False).circonscriptions_gagnees.sum()
        annual = pd.concat([annual, annual_2026], ignore_index=True)
        regions = pd.concat([regions, regions_2026[["annee", "region", "taux_moyen"]]], ignore_index=True)
        parties = pd.concat([parties, parties_2026[["annee", "parti", "voix", "part_voix"]]], ignore_index=True)
        wins = pd.concat([wins, wins_2026], ignore_index=True)

    available_years = sorted(annual.annee.astype(int).unique())
    with st.sidebar:
        st.markdown("## Paramètres")
        years = st.multiselect("Scrutins", available_years, default=available_years, key="legislative_years")
    if not years:
        st.info("Choisissez au moins un scrutin pour afficher les résultats.")
        return

    latest_year = max(years)
    selected = annual[annual.annee.isin(years)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Participation moyenne", f"{selected.taux_moyen.mean():.1%}")
    latest_count = int(annual.loc[annual.annee.eq(latest_year), "n_circonscriptions"].iloc[0])
    c2.metric(f"Circonscriptions · {latest_year}", f"{latest_count}")
    latest_wins = wins[wins.annee.eq(latest_year)]
    if latest_wins.empty:
        leading_party = "—"
    else:
        leader = latest_wins.loc[latest_wins.circonscriptions_gagnees.idxmax()]
        leading_party = f"{leader.parti_en_tete} · {int(leader.circonscriptions_gagnees)}"
    c3.metric(f"Parti le plus souvent en tête · {latest_year}", leading_party)

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Participation au fil du temps")
            trend = selected.copy()
            trend["annee"] = trend.annee.astype(str)
            fig = px.line(
                trend,
                x="annee",
                y="taux_moyen",
                markers=True,
                labels={"annee": "Scrutin", "taux_moyen": "Participation moyenne"},
                color_discrete_sequence=[TEAL],
            )
            fig.update_traces(line_width=3, marker_size=10)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_style(fig, 335), use_container_width=True)
    with right:
        with st.container(border=True):
            st.subheader(f"Participation par région · {latest_year}")
            regional = regions[regions.annee.eq(latest_year)].sort_values("taux_moyen")
            fig = px.bar(
                regional,
                x="taux_moyen",
                y="region",
                orientation="h",
                labels={"taux_moyen": "Participation moyenne", "region": "Région"},
                color="taux_moyen",
                color_continuous_scale=["#d6e8e3", TEAL],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(chart_style(fig, 335), use_container_width=True)

    parties_selected = parties[parties.annee.isin(years)].copy()
    top = parties_selected.groupby("parti").voix.sum().nlargest(8).index
    parties_selected = parties_selected[parties_selected.parti.isin(top)]
    parties_selected["annee"] = parties_selected.annee.astype(str)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Parts de voix des listes locales")
            fig = px.bar(
                parties_selected,
                x="parti",
                y="part_voix",
                color="annee",
                barmode="group",
                labels={"parti": "Parti", "part_voix": "Part des voix", "annee": "Scrutin"},
                color_discrete_sequence=[TEAL, GOLD, "#657A9A"],
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_style(fig, 350), use_container_width=True)
    with right:
        with st.container(border=True):
            st.subheader("Nombre de circonscriptions gagnées")
            wins_selected = wins[wins.annee.isin(years)].copy()
            wins_selected["annee"] = wins_selected.annee.astype(str)
            fig = px.bar(
                wins_selected,
                x="annee",
                y="circonscriptions_gagnees",
                color="parti_en_tete",
                barmode="stack",
                labels={"annee": "Scrutin", "circonscriptions_gagnees": "Circonscriptions", "parti_en_tete": "Parti"},
                color_discrete_sequence=[TEAL, GOLD, "#657A9A", "#CC7463", "#8A7CB8"],
            )
            st.plotly_chart(chart_style(fig, 350), use_container_width=True)

    if latest_year == 2021:
        st.subheader("Part des élues par région · 2021")
        women = profile[["region", "sieges", "repPctFemmes"]].sort_values("repPctFemmes")
        fig = px.bar(
            women,
            x="repPctFemmes",
            y="region",
            orientation="h",
            hover_data={"sieges": True, "repPctFemmes": ":.1%"},
            labels={"repPctFemmes": "Part des élues", "region": "Région", "sieges": "Sièges"},
            color_discrete_sequence=[GOLD],
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig, 400), use_container_width=True)
    elif latest_year == 2026:
        st.info("Le profil détaillé des élus 2026 n’est pas encore fourni par l’export utilisé.")

    st.markdown(
        '<div class="source-note">Les taux présentés sont des moyennes non pondérées par les inscrits. '
        'Les parts de voix portent sur les listes locales. Les résultats agrégés ne mesurent pas les intentions de vote. '
        'Les classeurs législatifs ne fournissent pas de géométries communales; leurs comparaisons sont présentées par région.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    with st.sidebar:
        st.markdown("# Maroc · Élections")
        st.caption("Données et analyse électorale")
        analysis = st.radio("Choisir une analyse", ["Communales", "Législatives"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Mini-projet · Programmation avancée pour l’IA")
    try:
        if analysis == "Communales":
            show_communal_dashboard()
        else:
            show_legislative_dashboard()
    except FileNotFoundError as exc:
        st.error("Les résultats ne sont pas encore générés.")
        st.code("python Script/run_pipeline.py", language="bash")
        st.caption(f"Fichier manquant : {Path(exc.filename or '').name}")


if __name__ == "__main__":
    main()
