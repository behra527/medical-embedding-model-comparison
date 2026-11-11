
from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIGURES = OUT / "figures"
DATASET = ROOT / "data" / "medical_sentence_pairs.csv"
REQUIRED_FILES = {
    "Sentence-pair results": OUT / "results.csv",
    "Overall model summary": OUT / "overall_summary.csv",
    "Category comparison": OUT / "category_comparison.csv",
}
MODEL_COLORS = {"General MiniLM": "#5965E8", "Biomedical PubMedBERT": "#16A6A0"}
CATEGORY_COLORS = {
    "synonyms": "#5965E8",
    "abbreviations": "#16A6A0",
    "negation": "#E99350",
    "numerical_difference": "#A879E5",
}
PAGES = ["Overview", "Model comparison", "Sentence explorer", "Visualizations", "Reports & exports"]

st.set_page_config(
    page_title="ClinEmbed | Model Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- REFINED LIGHT SAAS UI / UX DESIGN SYSTEM & SIDEBAR FIX ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #F4F6FB;
    --bg-card: #FFFFFF;
    --border-color: #E2E8F0;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --accent-indigo: #5965E8;
    --accent-cyan: #16A6A0;
}

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.stApp {
    background: var(--bg-main);
    color: var(--text-main);
}

.block-container {
    max-width: 1520px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 2.8rem;
    padding-right: 2.8rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* --- FIXED SIDEBAR VISIBILITY & WIDTH --- */
[data-testid="stSidebar"] {
    display: block !important;
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    min-width: 280px !important;
    max-width: 320px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

[data-testid="stSidebar"] * {
    color: #1E293B !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64748B !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.4rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.75rem 1rem !important;
    border-radius: 0.85rem !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #F8FAFC !important;
    border-color: #E2E8F0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #EEF2FF !important;
    border-color: #C7D2FE !important;
    box-shadow: 0 2px 8px rgba(89, 101, 232, 0.1);
}

[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: #3730A3 !important;
    font-weight: 700 !important;
}

/* Typography & Headings */
h1, h2, h3 {
    color: #0F172A !important;
    letter-spacing: -0.035em;
}
h1 { font-weight: 800 !important; font-size: 2.1rem !important; }
h2 { font-size: 1.3rem !important; font-weight: 750 !important; }
h3 { font-size: 1.1rem !important; font-weight: 700 !important; }

/* Cards & Containers */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}
[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #0F172A; font-size: 1.8rem; font-weight: 800; }

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: #FFFFFF;
    border-color: #E2E8F0;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.04);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
}

/* Buttons */
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: linear-gradient(135deg, #5965E8, #434CE6);
    border: none;
    border-radius: 12px;
    font-weight: 700;
    color: white;
    box-shadow: 0 4px 14px rgba(89, 101, 232, 0.3);
}
.stButton button, .stDownloadButton button {
    border-radius: 12px;
    font-weight: 650;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #1E293B;
}
.stButton button:hover { background: #F8FAFC; border-color: #94A3B8; }

[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    border-radius: 12px !important;
    background-color: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
}

/* Custom UI Blocks */
.brand-wrap { display: flex; align-items: center; gap: 14px; margin: 0 0 28px; }
.brand-icon {
    width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #5965E8, #16A6A0); color: white; font-size: 1.4rem; box-shadow: 0 8px 20px rgba(89, 101, 232, 0.25);
}
.brand-name { font-size: 1.2rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.03em; color: #0F172A; }
.brand-sub { font-size: 0.75rem; color: #64748B; letter-spacing: 0.03em; margin-top: 4px; font-weight: 600; }

.side-label { text-transform: uppercase; letter-spacing: 0.12em; color: #64748B; font-size: 0.7rem; font-weight: 800; margin-bottom: 12px; }
.side-card { border: 1px solid #E2E8F0; border-radius: 14px; background: #F8FAFC; padding: 16px; margin-top: 24px; }
.side-card .small { color: #64748B; font-size: 0.75rem; }
.side-card .large { color: #0F172A; font-size: 0.95rem; font-weight: 700; margin: 6px 0; }

.topline { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 800; color: #5965E8; margin-bottom: 10px; }
.page-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
.page-head h1 { margin: 0 0 8px; line-height: 1.2; }
.page-sub { color: #64748B; font-size: 0.95rem; max-width: 780px; line-height: 1.7; }

.live-pill {
    display: inline-flex; align-items: center; gap: 8px; border: 1px solid #A7F3D0;
    background: #ECFDF5; color: #047857; padding: 8px 14px; border-radius: 99px; font-size: 0.75rem; font-weight: 800; white-space: nowrap;
}
.live-dot { width: 7px; height: 7px; background: #10B981; display: inline-block; border-radius: 50%; box-shadow: 0 0 6px #10B981; }

.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 60%, #4338CA 100%);
    border-radius: 20px; padding: 32px 36px; color: white; margin-bottom: 26px;
    position: relative; overflow: hidden; box-shadow: 0 20px 40px rgba(67, 56, 202, 0.15);
}
.hero-kicker { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 800; color: #C7D2FE; margin-bottom: 12px; }
.hero-title { color: #FFFFFF; font-weight: 800; font-size: 1.6rem; line-height: 1.3; margin-bottom: 10px; letter-spacing: -0.03em; }
.hero-copy { max-width: 650px; font-size: 0.9rem; line-height: 1.7; color: #E0E7FF; }
.hero-tags { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 22px; }
.hero-tag { padding: 6px 12px; border-radius: 99px; font-size: 0.72rem; color: #C7D2FE; background: rgba(255, 255, 255, 0.12); border: 1px solid rgba(255, 255, 255, 0.2); font-weight: 600; }

.kpi-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 22px; min-height: 128px; }
.kpi-top { display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 0.78rem; font-weight: 650; }
.kpi-icon { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 9px; background: #EEF2FF; color: #5965E8; }
.kpi-value { color: #0F172A; font-size: 1.75rem; font-weight: 800; padding-top: 14px; }
.kpi-value.small { font-size: 1.05rem; }
.kpi-note { color: #94A3B8; font-size: 0.72rem; margin-top: 4px; }

.section-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; margin: 32px 0 14px; }
.section-head h3 { margin: 0; font-size: 1.1rem; }
.section-head p { margin: 4px 0 0; color: #64748B; font-size: 0.83rem; }
.tag { display: inline-flex; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 6px; padding: 4px 8px; font-size: 0.7rem; font-weight: 750; color: #4338CA; }

.category-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px; min-height: 130px; }
.category-name { color: #64748B; font-size: 0.78rem; font-weight: 700; margin: 0 0 10px; text-transform: uppercase; }
.category-winner { font-size: 0.95rem; font-weight: 800; color: #0F172A; margin-bottom: 8px; display: flex; align-items: center; }
.category-score { color: #64748B; font-size: 0.75rem; font-weight: 600; }

.dot-general { display: inline-block; width: 8px; height: 8px; background: #5965E8; border-radius: 50%; margin-right: 8px; }
.dot-biomedical { display: inline-block; width: 8px; height: 8px; background: #16A6A0; border-radius: 50%; margin-right: 8px; }

.sentence-text { border-radius: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 18px; color: #1E293B; line-height: 1.7; font-size: 0.88rem; min-height: 95px; }
.safe-footer { margin-top: 40px; padding: 24px 0 10px; color: #94A3B8; border-top: 1px solid #E2E8F0; font-size: 0.75rem; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt_num(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


def pretty_category(category: str) -> str:
    return str(category).replace("_", " ").title()


def short_winner(value: str) -> str:
    if value == "General MiniLM":
        return "General MiniLM"
    if value == "Biomedical PubMedBERT":
        return "Biomedical PubMedBERT"
    return "Tie"


def section(title: str, subtitle: str = "", pill: str | None = None) -> None:
    pill_html = f'<span class="tag">{escape(pill)}</span>' if pill else ""
    st.markdown(
        f'<div class="section-head"><div><h3>{escape(title)}</h3>'
        f'<p>{escape(subtitle)}</p></div>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def kpi(title: str, value: object, hint: str, symbol: str = "◈", small: bool = False) -> None:
    st.markdown(
        '<div class="kpi-card">'
        f'<div class="kpi-top"><span>{escape(str(title))}</span>'
        f'<span class="kpi-icon">{escape(symbol)}</span></div>'
        f'<div class="kpi-value{" small" if small else ""}">{escape(str(value))}</div>'
        f'<div class="kpi-note">{escape(str(hint))}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(section_name: str, title: str, subtitle: str, loaded: bool) -> None:
    badge = '<span class="live-pill"><span class="live-dot"></span> Analysis ready</span>' if loaded else ""
    st.markdown(
        f'<div class="topline">RESEARCH WORKSPACE / {escape(section_name.upper())}</div>'
        f'<div class="page-head"><div><h1>{escape(title)}</h1>'
        f'<div class="page-sub">{escape(subtitle)}</div></div>{badge}</div>',
        unsafe_allow_html=True,
    )


def fingerprint() -> tuple[int, ...]:
    return tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in REQUIRED_FILES.values())


@st.cache_data(show_spinner=False)
def load_results(_fingerprint: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(REQUIRED_FILES["Sentence-pair results"])
    overall = pd.read_csv(REQUIRED_FILES["Overall model summary"])
    categories = pd.read_csv(REQUIRED_FILES["Category comparison"])
    return results, overall, categories


def get_leader(overall: pd.DataFrame) -> tuple[str, float]:
    max_auc = float(overall["overall_roc_auc"].max())
    winners = overall[np.isclose(overall["overall_roc_auc"], max_auc)]
    return (str(winners.iloc[0]["model"]) if len(winners) == 1 else "Tie", max_auc)


def category_winner_cards(categories: pd.DataFrame) -> None:
    cols = st.columns(min(len(categories), 4))
    for i, row in enumerate(categories.itertuples(index=False)):
        with cols[i % len(cols)]:
            winner = short_winner(str(row.higher_category_auc))
            value = max(float(row.general_category_auc), float(row.biomedical_category_auc))
            dot = "dot-biomedical" if winner == "Biomedical PubMedBERT" else "dot-general"
            st.markdown(
                '<div class="category-card">'
                f'<div class="category-name">{escape(pretty_category(row.category))}</div>'
                f'<div class="category-winner"><span class="{dot}"></span>{escape(winner)}</div>'
                f'<div class="category-score">Top category ranking AUC &nbsp; {fmt_num(value)}</div>'
                '</div>', unsafe_allow_html=True,
            )


def style_chart(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(stroke=None).configure_axis(
        labelColor="#64748B", titleColor="#475569", gridColor="#F1F5F9", domain=False,
        tickColor="#CBD5E1", labelFontSize=11, titleFontSize=11,
    ).configure_legend(labelColor="#475569", labelFontSize=11).configure(background="#FFFFFF")


def auc_chart(categories: pd.DataFrame, height: int = 290) -> alt.Chart:
    source = categories.melt(
        id_vars=["category"],
        value_vars=["general_category_auc", "biomedical_category_auc"],
        var_name="model", value_name="auc",
    )
    source["model"] = source["model"].map({
        "general_category_auc": "General MiniLM", "biomedical_category_auc": "Biomedical PubMedBERT",
    })
    source["category_name"] = source["category"].map(pretty_category)
    order = [pretty_category(c) for c in categories["category"]]
    chart = alt.Chart(source).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("category_name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
        xOffset=alt.XOffset("model:N", sort=list(MODEL_COLORS)),
        y=alt.Y("auc:Q", title="Category ranking AUC", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title=None,
            scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
            legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("category_name:N", title="Category"), "model:N", alt.Tooltip("auc:Q", format=".3f")],
    ).properties(height=height)
    return style_chart(chart)


def scatter_chart(results: pd.DataFrame, height: int = 340) -> alt.Chart:
    plot = results.copy()
    plot["category_name"] = plot["category"].map(pretty_category)
    plot["equivalence"] = np.where(plot["same_meaning"] == 1, "Equivalent", "Different")
    base = alt.Chart(plot).mark_circle(size=110, opacity=0.8, stroke="#FFFFFF", strokeWidth=1.2).encode(
        x=alt.X("general_cosine:Q", title="MiniLM cosine similarity", scale=alt.Scale(domain=[-1, 1])),
        y=alt.Y("biomedical_cosine:Q", title="PubMedBERT cosine similarity", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("category:N", title="Sentence category",
            scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
            legend=alt.Legend(orient="top", labelExpr="replace(datum.label, '_', ' ')")),
        tooltip=["pair_id:N", "category_name:N", "equivalence:N", "sentence_1:N", "sentence_2:N",
                 alt.Tooltip("general_cosine:Q", format=".3f"), alt.Tooltip("biomedical_cosine:Q", format=".3f")],
    )
    line = alt.Chart(pd.DataFrame({"x": [-1, 1], "y": [-1, 1]})).mark_line(
        stroke="#94A3B8", strokeDash=[5, 5], strokeWidth=1.5,
    ).encode(x="x:Q", y="y:Q")
    return style_chart((base + line).properties(height=height))


def score_histogram(results: pd.DataFrame, height: int = 280) -> alt.Chart:
    plot = results.melt(id_vars=["pair_id", "same_meaning"],
        value_vars=["general_cosine", "biomedical_cosine"], var_name="model", value_name="cosine")
    plot["model"] = plot["model"].map({"general_cosine": "General MiniLM", "biomedical_cosine": "Biomedical PubMedBERT"})
    plot["label"] = np.where(plot["same_meaning"] == 1, "Equivalent", "Different")
    chart = alt.Chart(plot).mark_bar(opacity=0.85).encode(
        x=alt.X("cosine:Q", bin=alt.Bin(maxbins=18), title="Cosine similarity", scale=alt.Scale(domain=[-1, 1])),
        y=alt.Y("count():Q", title="Pairs"),
        color=alt.Color("label:N", scale=alt.Scale(domain=["Equivalent", "Different"], range=["#16A6A0", "#E99350"]), title=None),
        column=alt.Column("model:N", title=None, sort=list(MODEL_COLORS), header=alt.Header(labelFontSize=12, labelFontWeight="bold")),
        tooltip=["model:N", "label:N", alt.Tooltip("count():Q", title="Pairs")],
    ).properties(height=height)
    return style_chart(chart)


def overview(results: pd.DataFrame, overall: pd.DataFrame, categories: pd.DataFrame) -> None:
    page_header("Overview", "Embedding intelligence, simplified.",
                "Your medical semantic similarity benchmark, all in one research workspace.", True)
    leader, top_auc = get_leader(overall)
    st.markdown(
        '<div class="hero"><div class="hero-kicker">CLINEMBED / EXPERIMENT DASHBOARD</div>'
        '<div class="hero-title">Understand medical language.<br>Compare model behaviour.</div>'
        '<div class="hero-copy">Explore semantic equivalence, domain-specific performance, and differences'
        ' in embedding spaces using real model outputs from your analysis pipeline.</div>'
        '<div class="hero-tags"><span class="hero-tag">↗ Cosine similarity</span>'
        '<span class="hero-tag">◎ Clinical sentence pairs</span>'
        '<span class="hero-tag">◈ PCA visualization</span></div></div>',
        unsafe_allow_html=True,
    )
    cells = st.columns(4, gap="medium")
    with cells[0]: kpi("Sentence pairs", f"{len(results):,}", "Evaluated from results.csv", "↗")
    with cells[1]: kpi("Embedding models", len(overall), "General & biomedical", "◉")
    with cells[2]: kpi("Best overall AUC", fmt_num(top_auc), "Ranking quality · higher is better", "◈")
    with cells[3]: kpi("Leading model", leader, "Based on overall ranking AUC", "✦", small=True)

    section("Performance by category", "AUC compares each category with opposite-label sentence pairs.", "MODEL BENCHMARK")
    with st.container(border=True):
        st.altair_chart(auc_chart(categories), width="stretch", theme=None)
    section("Category leaders", "Winners are calculated from category ranking AUC, not raw cosine scores.")
    category_winner_cards(categories)

    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        section("Similarity agreement", "Each point is one evaluated sentence pair.")
        with st.container(border=True):
            st.altair_chart(scatter_chart(results, 300), width="stretch", theme=None)
    with right:
        section("Model summary", "Real evaluation metrics from the analysis run.")
        table = overall[["model", "overall_roc_auc", "separation_gap", "embedding_dimensions"]].copy()
        table.columns = ["Model", "Ranking AUC", "Separation gap", "Dimensions"]
        st.dataframe(table, hide_index=True, width="stretch",
            column_config={"Ranking AUC": st.column_config.NumberColumn(format="%.3f"),
                           "Separation gap": st.column_config.NumberColumn(format="%.3f")})


def model_comparison(results: pd.DataFrame, overall: pd.DataFrame, categories: pd.DataFrame) -> None:
    page_header("Benchmark", "Model performance", "Side-by-side evaluation of general-purpose and biomedical sentence embeddings.", True)
    cells = st.columns(2, gap="medium")
    for i, row in enumerate(overall.itertuples(index=False)):
        with cells[i % 2]:
            name = str(row.model)
            description = "General-purpose semantic embeddings" if "General" in name else "Domain-trained biomedical embeddings"
            st.markdown(f'<div class="topline">{escape(description.upper())}</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"### {escape(name)}")
                st.caption(str(row.model_id))
                a, b = st.columns(2)
                a.metric("Ranking AUC", fmt_num(row.overall_roc_auc))
                b.metric("Separation gap", fmt_num(row.separation_gap))
                st.caption(f"Vector size: {int(row.embedding_dimensions)} dimensions")
    section("Category-level AUC", "Higher is better; 0.5 represents chance-level ranking.")
    with st.container(border=True):
        st.altair_chart(auc_chart(categories, 350), width="stretch", theme=None)
    section("Similarity distributions", "Equivalent versus non-equivalent examples in each model space.")
    with st.container(border=True):
        st.altair_chart(score_histogram(results), width="stretch", theme=None)
    section("All benchmark metrics", "Exact output values, not approximations.")
    st.dataframe(overall, hide_index=True, width="stretch")


def sentence_explorer(results: pd.DataFrame) -> None:
    page_header("Data explorer", "Sentence pair explorer", "Search, filter and inspect the exact medical passages evaluated by both models.", True)
    c1, c2, c3 = st.columns([1.4, 1, 1], gap="small")
    with c1: search = st.text_input("Search passages", placeholder="Search sentence text, pair ID or medical term...")
    with c2:
        choices = ["All categories"] + sorted(results.category.unique().tolist())
        chosen = st.selectbox("Category", choices, format_func=lambda x: pretty_category(x) if x != "All categories" else x)
    with c3: label = st.selectbox("Gold label", ["All labels", "Equivalent", "Different"])
    
    filtered = results.copy()
    if search.strip():
        q = search.strip()
        mask = (filtered["sentence_1"].str.contains(q, case=False, regex=False, na=False) |
                filtered["sentence_2"].str.contains(q, case=False, regex=False, na=False) |
                filtered["pair_id"].astype(str).str.contains(q, case=False, regex=False, na=False))
        filtered = filtered.loc[mask]
    if chosen != "All categories":
        filtered = filtered[filtered["category"] == chosen]
    if label != "All labels":
        filtered = filtered[filtered["same_meaning"] == (1 if label == "Equivalent" else 0)]

    section("Filtered evaluation data", f"Showing {len(filtered)} of {len(results)} real pairs.", "INTERACTIVE TABLE")
    display = filtered[["pair_id", "category", "sentence_1", "sentence_2", "same_meaning", "general_cosine", "biomedical_cosine"]].copy()
    display["category"] = display["category"].map(pretty_category)
    display["same_meaning"] = display["same_meaning"].map({1: "Equivalent", 0: "Different"})
    display.columns = ["Pair ID", "Category", "Sentence A", "Sentence B", "Gold label", "MiniLM", "PubMedBERT"]
    st.dataframe(display, width="stretch", hide_index=True, height=390)


def visualizations(results: pd.DataFrame) -> None:
    page_header("Visual analytics", "Embedding visualizations", "Explore PCA projections and saved diagnostics from the evaluated models.", True)
    st.info("Ensure generated figures exist in your `outputs/figures/` folder to view them here.")
    if FIGURES.exists():
        figs = list(FIGURES.glob("*.png"))
        if figs:
            for fig_path in figs:
                st.image(str(fig_path), caption=fig_path.stem.replace("_", " ").title(), width="stretch")
        else:
            st.warning("No saved figures found in outputs/figures.")


def reports_and_exports(results: pd.DataFrame, overall: pd.DataFrame, categories: pd.DataFrame) -> None:
    page_header("Reports & Exports", "Download Evaluation Artifacts", "Export summary tables, model benchmarks, and results as a ZIP bundle.", True)
    
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr("results.csv", results.to_csv(index=False))
        zip_file.writestr("overall_summary.csv", overall.to_csv(index=False))
        zip_file.writestr("category_comparison.csv", categories.to_csv(index=False))
    
    st.download_button(
        "↓ Download Complete ZIP Bundle",
        data=buf.getvalue(),
        file_name="clinembed_evaluation_bundle.zip",
        mime="application/zip",
        width="stretch",
    )


def main() -> None:
    try:
        results, overall, categories = load_results(fingerprint())
    except Exception as e:
        st.error(f"Failed to load required analysis files from `outputs/`: {e}")
        st.stop()

    with st.sidebar:
        st.markdown(
            '<div class="brand-wrap"><div class="brand-icon">🧬</div>'
            '<div><div class="brand-name">ClinEmbed</div>'
            '<div class="brand-sub">Model Intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
        chosen_page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.markdown('<div class="side-card"><div class="small">Workspace status</div>'
                    '<div class="large">Outputs Verified</div>'
                    '<div class="small">All CSVs loaded successfully.</div></div>', unsafe_allow_html=True)

    if chosen_page == "Overview":
        overview(results, overall, categories)
    elif chosen_page == "Model comparison":
        model_comparison(results, overall, categories)
    elif chosen_page == "Sentence explorer":
        sentence_explorer(results)
    elif chosen_page == "Visualizations":
        visualizations(results)
    elif chosen_page == "Reports & exports":
        reports_and_exports(results, overall, categories)

    st.markdown('<div class="safe-footer">ClinEmbed Research Analytics · Built for Semantic Embedding Benchmarks</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()