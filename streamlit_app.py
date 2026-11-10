"""Streamlit dashboard for the Data-Driven Social Engagement Initiative.

Reads ONLY the CSV/JSON files exported by the notebook (datasets/processed, reports).
Place this file in the project root (next to the datasets/ folder) and run:
    streamlit run streamlit_app.py
It also works if kept in dashboard/ (the project root is detected automatically).
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "datasets").exists() else HERE.parent
PROC = ROOT / "datasets" / "processed"
st.set_page_config(page_title="Social Engagement Dashboard", layout="wide")

@st.cache_data
def load(name, dates=()):
    path = PROC / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {PROC}. Run the notebook first.")
    return pd.read_csv(path, parse_dates=list(dates))

manifest_path = ROOT / "reports" / "project_manifest.json"
mode = json.loads(manifest_path.read_text())["data_mode"] if manifest_path.exists() else "UNKNOWN"
st.title("Data-Driven Social Engagement Initiative")
if mode != "REAL":
    st.warning(f"Data mode: {mode}. Figures are NOT real social-media results.")

posts = load("content_analysis_72h.csv", ["publish_date"])
comments = load("comments_scored.csv", ["publish_date", "comment_date"])
curve = load("save_share_curve.csv")
recs = load("recommendations.csv")
weekly_plan = load("weekly_plan.csv")
trend_data = load("trend_data.csv", ["observation_date"])
forecast = load("trend_forecast.csv")
trend_sum = load("trend_summary.csv")
scorecard = load("topic_scorecard.csv")
metrics = load("model_metrics.csv")
ab = load("ab_results.csv")

st.sidebar.header("Filters")
platforms = st.sidebar.multiselect("Platform", sorted(posts["platform"].unique()), default=sorted(posts["platform"].unique()))
topics = st.sidebar.multiselect("Topic", sorted(posts["topic"].unique()), default=sorted(posts["topic"].unique()))
formats = st.sidebar.multiselect("Content format", sorted(posts["content_type"].unique()), default=sorted(posts["content_type"].unique()))
dmin, dmax = posts["publish_date"].min().date(), posts["publish_date"].max().date()
span = st.sidebar.date_input("Publish date range", (dmin, dmax))
start, end = (span[0], span[1]) if len(span) == 2 else (dmin, dmax)

def flt(df, date_col="publish_date"):
    m = df["platform"].isin(platforms) & df["topic"].isin(topics) & df["content_type"].isin(formats)
    return df[m & df[date_col].dt.date.between(start, end)]

v, c = flt(posts), flt(comments)
if v.empty:
    st.error("No posts match the filters.")
    st.stop()

tabs = st.tabs(["Overview", "Topics", "Sentiment", "Save-to-Share", "Recommendations", "Trends", "Models and A/B"])
with tabs[0]:
    a, b, d = st.columns(3)
    a.metric("Posts", len(v)); b.metric("Mean VES z", f"{v['ves_core_z'].mean():.2f}"); d.metric("Followers gained", int(v["followers_gained"].sum()))
    wk = v.assign(week=v["publish_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()).groupby(["week", "platform"])["ves_core_z"].mean().unstack()
    st.subheader("Weekly engagement (VES z-score)"); st.line_chart(wk)
with tabs[1]:
    t = v.groupby("topic").agg(posts=("content_id", "size"), mean_ves_z=("ves_core_z", "mean"), followers_per_1k=("followers_per_1k_exposure", "mean")).sort_values("mean_ves_z", ascending=False)
    st.bar_chart(t["mean_ves_z"]); st.dataframe(t); st.subheader("Topic scorecard (all data)"); st.dataframe(scorecard)
with tabs[2]:
    if c.empty:
        st.info("No comments for this filter.")
    else:
        st.subheader("Sentiment by topic"); st.bar_chart(pd.crosstab(c["topic"], c["pred_sentiment"], normalize="index"))
        st.subheader("Relatable-comment share by topic"); st.bar_chart(c.groupby("topic")["is_relatable"].mean())
with tabs[3]:
    ig = curve[curve["topic"].isin(topics)]
    if ig.empty:
        st.info("Save-to-share exists for Instagram only.")
    else:
        st.line_chart(ig.pivot(index="hours_since_publish", columns="topic", values="interval_save_to_share"))
with tabs[4]:
    plat = st.sidebar.selectbox("Recommendations for platform", sorted(recs["platform"].unique()))
    st.subheader("Plan for the coming week"); st.dataframe(weekly_plan)
    st.subheader(f"Top 20 candidates on {plat}"); st.dataframe(recs[recs["platform"] == plat].head(20))
with tabs[5]:
    term = st.selectbox("Keyword / hashtag", sorted(trend_data["term"].unique()))
    st.line_chart(trend_data[trend_data["term"] == term].set_index("observation_date")["trend_score"])
    st.subheader("Next-7-day forecast"); st.dataframe(forecast); st.subheader("Detected trends (past)"); st.dataframe(trend_sum)
with tabs[6]:
    st.subheader("Model comparison (chronological validation)"); st.dataframe(metrics)
    st.subheader("A/B results (observational)"); st.dataframe(ab)
