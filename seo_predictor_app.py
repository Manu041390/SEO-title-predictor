# seo_predictor_app.py

import streamlit as st
import pandas as pd
import datetime

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from prophet import Prophet
from prophet.plot import plot_plotly

# ─── CONFIG ────────────────────────────────────────────────────────────────
GSC_KEYFILE = "searchconsole-455516-f95aab4b66a5.json"
GSC_SCOPES  = ["https://www.googleapis.com/auth/webmasters.readonly"]
SITE_URL    = "https://locusit.se/"
MAX_HISTORY = 90

# ─── GSC API HELPERS ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def gsc_service():
    creds = Credentials.from_service_account_file(GSC_KEYFILE, scopes=GSC_SCOPES)
    return build("searchconsole", "v1", credentials=creds)

@st.cache_data(ttl=3600)
def fetch_site_data(days=MAX_HISTORY):
    svc   = gsc_service()
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days)
    body  = {"startDate": str(start), "endDate": str(today), "dimensions": ["date"], "rowLimit": 10000}
    resp  = svc.searchanalytics().query(siteUrl=SITE_URL, body=body).execute()
    rows  = resp.get("rows", [])
    return pd.DataFrame({
        "ds": pd.to_datetime([r["keys"][0] for r in rows]),
        "y":  [r.get("clicks", 0) for r in rows]
    }).sort_values("ds")

@st.cache_data(ttl=3600)
def fetch_queries_df(days=30, limit=10):
    svc   = gsc_service()
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days)
    body  = {
        "startDate":   str(start),
        "endDate":     str(today),
        "dimensions":  ["query"],
        "rowLimit":    limit,
        "metrics":     ["clicks","impressions"]
    }
    resp = svc.searchanalytics().query(siteUrl=SITE_URL, body=body).execute()
    return pd.DataFrame([{
        "keyword":     r["keys"][0],
        "clicks":      r.get("clicks",0),
        "impressions": r.get("impressions",0)
    } for r in resp.get("rows",[])])

@st.cache_data(ttl=3600)
def fetch_page_data(path, days=MAX_HISTORY):
    svc   = gsc_service()
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days)
    body  = {
        "startDate": str(start),
        "endDate":   str(today),
        "dimensions": ["date"],
        "dimensionFilterGroups": [{
            "filters": [{
                "dimension":  "page",
                "operator":   "equals",
                "expression": SITE_URL.rstrip("/") + path
            }]
        }],
        "rowLimit": 10000
    }
    resp = svc.searchanalytics().query(siteUrl=SITE_URL, body=body).execute()
    rows = resp.get("rows", [])
    return pd.DataFrame({
        "ds": pd.to_datetime([r["keys"][0] for r in rows]),
        "y":  [r.get("clicks", 0) for r in rows]
    }).sort_values("ds")

@st.cache_data(ttl=3600)
def fetch_keyword_data(keyword, days=MAX_HISTORY):
    svc   = gsc_service()
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days)
    body  = {
        "startDate": str(start),
        "endDate":   str(today),
        "dimensions": ["date"],
        "dimensionFilterGroups": [{
            "filters": [{
                "dimension":  "query",
                "operator":   "equals",
                "expression": keyword
            }]
        }],
        "rowLimit": 10000
    }
    resp = svc.searchanalytics().query(siteUrl=SITE_URL, body=body).execute()
    rows = resp.get("rows", [])
    return pd.DataFrame({
        "ds": pd.to_datetime([r["keys"][0] for r in rows]),
        "y":  [r.get("clicks", 0) for r in rows]
    }).sort_values("ds")

# ─── FORECASTING ───────────────────────────────────────────────────────────
def forecast(df, days_ahead=15):
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=days_ahead)
    fc     = m.predict(future)
    return m, fc

# ─── SEO HEURISTIC ────────────────────────────────────────────────────────
def keyword_in_title(title, kw):
    return kw.strip().lower() in title.strip().lower()

# ─── STREAMLIT APP ────────────────────────────────────────────────────────
st.set_page_config(page_title="🔎 SEO & GSC Analytics", layout="wide")
st.title("🔎 SEO & GSC Analytics Dashboard")

tabs = st.tabs(["SEO Prediction", "Keyword Ranking", "Traffic Projections", "Keyword Forecast"])

# Tab 1: SEO Prediction & 15-Day Site Forecast
with tabs[0]:
    st.header("📝 SEO Prediction & 15-Day Site-wide Forecast")
    title      = st.text_input("Blog Title")
    keyword    = st.text_input("Main Keyword")
    word_count = st.number_input("Word Count", min_value=100, max_value=5000, value=800)

    if st.button("Run Prediction & Forecast"):
        conf = 80 if keyword_in_title(title, keyword) else 50
        label = "Optimizable ✅" if conf > 70 else "Not Optimizable ❌"
        st.metric("SEO Prediction", label, delta=f"{conf}% confidence")

        if conf > 70:
            df_site = fetch_site_data()
            if df_site.empty:
                st.warning("No GSC data for your site.")
            else:
                total = int(df_site["y"].sum())
                avg   = round(df_site["y"].mean(), 2)
                last7 = df_site.tail(7)["y"].mean()
                prev7 = df_site.head(7)["y"].mean() or 1
                pct   = round((last7 - prev7)/prev7 * 100, 2)

                st.subheader("📊 Key Insights (Last 90 Days)")
                st.write(f"- Total clicks: **{total}**")
                st.write(f"- Avg daily clicks: **{avg}**")
                st.write(f"- 7-day vs prior 7-day: **{pct}%**")

                st.subheader("📈 15-Day Click Forecast (Entire Site)")
                m, fc = forecast(df_site, days_ahead=15)
                fig = plot_plotly(m, fc)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("💡 Alternative Keywords")
            df_q = fetch_queries_df(30, 5)
            for kw in df_q["keyword"]:
                st.write(f"- {kw}")

# Tab 2: Keyword Ranking
with tabs[1]:
    st.header("🔑 Top Query Ranking")
    d = st.number_input("History Days", 7, MAX_HISTORY, 30, key="rk_days")
    n = st.number_input("Top N", 5, 50, 10, key="rk_n")
    if st.button("Load Ranking", key="rk_run"):
        df_r = fetch_queries_df(days=d, limit=n)
        if df_r.empty:
            st.warning("No query data.")
        else:
            st.metric("Total Clicks", int(df_r["clicks"].sum()))
            st.metric("Avg CTR (%)", round((df_r["clicks"]/df_r["impressions"]).mean()*100,2))
            st.dataframe(df_r)
            st.bar_chart(df_r.set_index("keyword")[["clicks","impressions"]])

# Tab 3: Blog Forecast (30 days)
with tabs[2]:
    st.header("📈 30-Day Traffic Projections")
    path = st.text_input("Blog title ", key="pf_path")
    if st.button("Fetch & Forecast Page", key="pf_run"):
        dfp = fetch_page_data(path)
        if dfp.empty:
            st.warning("No data for that page.")
        else:
            st.line_chart(dfp.set_index("ds")["y"])
            m, fc = forecast(dfp, days_ahead=30)
            st.plotly_chart(plot_plotly(m, fc), use_container_width=True)

# Tab 4: Keyword Forecast (4 days)
with tabs[3]:
    st.header("🔮 4-Day Keyword Forecast")
    kw = st.text_input("Keyword to Forecast", key="kf_kw")
    if st.button("Fetch & Forecast Keyword", key="kf_run"):
        dfk = fetch_keyword_data(kw)
        if dfk.empty:
            st.warning("No data for that keyword.")
        else:
            st.line_chart(dfk.set_index("ds")["y"])
            m2, fc2 = forecast(dfk, days_ahead=4)
            st.plotly_chart(plot_plotly(m2, fc2), use_container_width=True)
