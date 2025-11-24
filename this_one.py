import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================

st.set_page_config(
    page_title="Number Pattern Finder (DR / SG / FB / GZ / GL)",
    layout="wide",
)

GROUP_COLS = ["DR", "SG", "FB", "GZ", "GL"]
YEAR_COL, MONTH_COL, DAY_COL = "Year", "Month", "Day"


# =========================
# Data loading & preparation
# =========================

@st.cache_data(show_spinner=True)
def load_data(file) -> pd.DataFrame:
    if file is None:
        # Fallback: try local file name (for your own environment)
        df = pd.read_csv("August - August.csv")
    else:
        df = pd.read_csv(file)

    # Basic sanity checks
    expected = set([YEAR_COL, MONTH_COL, DAY_COL]) | set(GROUP_COLS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build a proper datetime column and sort
    df = df.copy()
    df["date"] = pd.to_datetime(
        dict(year=df[YEAR_COL], month=df[MONTH_COL], day=df[DAY_COL]),
        errors="coerce",
    )
    df = df.sort_values("date").reset_index(drop=True)

    # Compute "days_ago" from the most recent date
    max_date = df["date"].max()
    df["days_ago"] = (max_date - df["date"]).dt.days

    # Convert group columns to numeric with NaN for "xx" or non-numeric
    for col in GROUP_COLS:
        df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_numeric_series(df: pd.DataFrame, group: str) -> pd.Series:
    """Return numeric series for a group with NaNs for missing/xx."""
    return df[group + "_num"]


# =========================
# Frequency & recency stats
# =========================

@st.cache_data(show_spinner=False)
def compute_frequency_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    s = get_numeric_series(df, group)
    max_date = df["date"].max()

    # Total frequency
    freq_total = s.value_counts(dropna=True)

    # Helper to compute windowed frequencies
    def freq_in_last(days: int) -> pd.Series:
        mask = df["days_ago"] <= days
        return s[mask].value_counts(dropna=True)

    freq_30 = freq_in_last(30)
    freq_60 = freq_in_last(60)
    freq_90 = freq_in_last(90)

    # Last seen date for each number
    last_seen_idx = df.loc[s.notna()].groupby(s.name)["date"].idxmax()
    last_seen_dates = df.loc[last_seen_idx, ["date"]]
    last_seen_dates.index = last_seen_idx.index  # index is number itself

    # Build table over numbers 0–99
    all_nums = pd.Index(range(100), name="number")
    tbl = pd.DataFrame(index=all_nums)

    tbl["total_freq"] = freq_total.reindex(all_nums).fillna(0).astype(int)
    tbl["freq_30"] = freq_30.reindex(all_nums).fillna(0).astype(int)
    tbl["freq_60"] = freq_60.reindex(all_nums).fillna(0).astype(int)
    tbl["freq_90"] = freq_90.reindex(all_nums).fillna(0).astype(int)

    # Last seen days ago
    last_seen = last_seen_dates["date"]
    tbl["last_seen_date"] = last_seen.reindex(all_nums)
    tbl["last_seen_days_ago"] = (
        (max_date - tbl["last_seen_date"]).dt.days
    ).astype("float")

    tbl["is_seen"] = tbl["total_freq"] > 0

    # Hot / Warm / Cold tags (simple heuristic on last 90 days)
    q80 = tbl["freq_90"].quantile(0.80)
    q50 = tbl["freq_90"].quantile(0.50)

    def tag_row(row):
        if row["total_freq"] == 0:
            return "Never"
        if row["freq_90"] >= q80 and row["last_seen_days_ago"] <= 15:
            return "Hot"
        if row["freq_90"] >= q50 and row["last_seen_days_ago"] <= 45:
            return "Warm"
        if pd.isna(row["last_seen_days_ago"]):
            return "Never"
        if row["last_seen_days_ago"] > 120:
            return "Cold"
        return "Neutral"

    tbl["tag"] = tbl.apply(tag_row, axis=1)

    return tbl.reset_index()


# =========================
# Gap / cycle stats
# =========================

@st.cache_data(show_spinner=False)
def compute_gap_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    s = get_numeric_series(df, group)
    dates = df["date"]

    rows = []
    for num in range(100):
        idx = s.index[s == num]
        if len(idx) == 0:
            rows.append(
                dict(
                    number=num,
                    appearances=0,
                    gaps_count=0,
                    mean_gap_days=np.nan,
                    median_gap_days=np.nan,
                    std_gap_days=np.nan,
                    min_gap_days=np.nan,
                    max_gap_days=np.nan,
                    rhythm="Never",
                )
            )
            continue

        # Compute day gaps between consecutive appearances
        if len(idx) >= 2:
            d = dates.loc[idx].values.astype("datetime64[D]").astype(int)
            gaps = np.diff(d)
            mean_gap = float(np.mean(gaps))
            median_gap = float(np.median(gaps))
            std_gap = float(np.std(gaps))
            min_gap = int(np.min(gaps))
            max_gap = int(np.max(gaps))
            gaps_count = len(gaps)
        else:
            gaps = np.array([])
            mean_gap = median_gap = std_gap = min_gap = max_gap = np.nan
            gaps_count = 0

        # Simple rhythm classification
        if len(idx) == 1:
            rhythm = "Single"
        elif gaps_count >= 3 and std_gap <= 7:
            rhythm = "Rhythmic"
        elif gaps_count >= 3 and std_gap <= 15:
            rhythm = "Semi-rhythmic"
        else:
            rhythm = "Irregular"

        rows.append(
            dict(
                number=num,
                appearances=len(idx),
                gaps_count=gaps_count,
                mean_gap_days=mean_gap,
                median_gap_days=median_gap,
                std_gap_days=std_gap,
                min_gap_days=min_gap,
                max_gap_days=max_gap,
                rhythm=rhythm,
            )
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_gaps_for_number(df: pd.DataFrame, group: str, number: int) -> Optional[pd.Series]:
    s = get_numeric_series(df, group)
    dates = df["date"]
    idx = s.index[s == number]
    if len(idx) < 2:
        return None

    d = dates.loc[idx].values.astype("datetime64[D]").astype(int)
    gaps = np.diff(d)
    return pd.Series(gaps, name="gap_days")


# =========================
# Co-occurrence stats
# =========================

@st.cache_data(show_spinner=False)
def compute_pair_stats(
    df: pd.DataFrame,
    base_group: str,
    target_group: str,
) -> pd.DataFrame:
    base = get_numeric_series(df, base_group)
    target = get_numeric_series(df, target_group)

    mask = base.notna() & target.notna()
    base = base[mask].astype(int)
    target = target[mask].astype(int)

    if base.empty:
        return pd.DataFrame(
            columns=[
                "base_number",
                "target_number",
                "pair_count",
                "support",
                "confidence",
                "lift",
            ]
        )

    total_rows = len(base)

    pair_df = pd.DataFrame({"base": base, "target": target})
    pair_counts = pair_df.value_counts().rename("pair_count").reset_index()

    base_counts = base.value_counts().rename("base_count")
    target_counts = target.value_counts().rename("target_count")

    # Join counts
    pair_counts["base_number"] = pair_counts["base"].astype(int)
    pair_counts["target_number"] = pair_counts["target"].astype(int)
    pair_counts = pair_counts.drop(columns=["base", "target"])

    pair_counts = pair_counts.merge(
        base_counts.rename_axis("base_number").reset_index(),
        on="base_number",
        how="left",
    )
    pair_counts = pair_counts.merge(
        target_counts.rename_axis("target_number").reset_index(),
        on="target_number",
        how="left",
    )

    pair_counts["support"] = pair_counts["pair_count"] / total_rows
    pair_counts["confidence"] = pair_counts["pair_count"] / pair_counts["base_count"]
    pair_counts["target_prob"] = pair_counts["target_count"] / total_rows
    pair_counts["lift"] = pair_counts["confidence"] / pair_counts["target_prob"]

    return pair_counts[
        [
            "base_number",
            "target_number",
            "pair_count",
            "support",
            "confidence",
            "lift",
            "base_count",
            "target_count",
        ]
    ].sort_values("lift", ascending=False)


# =========================
# Cluster stats
# =========================

@st.cache_data(show_spinner=False)
def compute_cluster_features(df: pd.DataFrame, group: str) -> pd.DataFrame:
    freq_tbl = compute_frequency_table(df, group)
    gap_tbl = compute_gap_table(df, group)

    merged = freq_tbl.merge(gap_tbl, on="number", how="left", suffixes=("", "_gap"))

    # Features for clustering
    features = merged[
        [
            "total_freq",
            "freq_90",
            "mean_gap_days",
            "std_gap_days",
            "last_seen_days_ago",
        ]
    ].copy()

    # Replace NaNs for unseen / single-appearance numbers
    features = features.fillna(
        {
            "mean_gap_days": 0.0,
            "std_gap_days": 0.0,
            "last_seen_days_ago": merged["last_seen_days_ago"].max()
            if not merged["last_seen_days_ago"].isna().all()
            else 0.0,
        }
    )

    features = features.astype(float)
    return merged, features


def run_clustering(
    merged: pd.DataFrame, features: pd.DataFrame, n_clusters: int
) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    # To avoid errors when there are too few distinct data points
    n_clusters = max(1, min(n_clusters, len(features)))
    if n_clusters == 1:
        merged["cluster"] = 0
        return merged

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    merged["cluster"] = kmeans.fit_predict(X_scaled)
    return merged


# =========================
# UI
# =========================

st.title("🔍 Number Pattern Finder — DR / SG / FB / GZ / GL")

st.markdown(
    """
This app explores **patterns of numbers** across the five groups:
**DR, SG, FB, GZ, GL**.

It is fully **unsupervised** and focuses on:
- Frequency & recency (hot / warm / cold),
- Gap / cycle behaviour,
- Cross-group co-occurrence,
- Behavioural clusters (via K-Means).
"""
)

st.sidebar.header("1️⃣ Load data")
uploaded = st.sidebar.file_uploader(
    "Upload your CSV (optional — otherwise uses `August - August.csv`)",
    type=["csv"],
)

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.success(
    f"Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}"
)

st.sidebar.header("2️⃣ Global filters")
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider(
    "Restrict to Year range (for all analysis)",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

mask_years = (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
df_filtered = df.loc[mask_years].copy()
if df_filtered.empty:
    st.error("No data in the selected year range.")
    st.stop()

st.sidebar.info(
    f"Using {len(df_filtered)} rows between years {year_range[0]}–{year_range[1]}."
)

st.markdown(
    f"**Active year filter:** {year_range[0]}–{year_range[1]}  "
    f"({len(df_filtered)} rows)"
)

group_tabs = st.tabs(GROUP_COLS)

for group, tab in zip(GROUP_COLS, group_tabs):
    with tab:
        st.subheader(f"Group: {group}")

        freq_tab, gap_tab, co_tab, cluster_tab = st.tabs(
            ["Frequency & Recency", "Gap / Cycle", "Co-occurrence", "Cluster View"]
        )

        # =============================
        # Frequency & recency tab
        # =============================
        with freq_tab:
            st.markdown("### Frequency & Recency (Hot / Warm / Cold)")

            freq_tbl = compute_frequency_table(df_filtered, group)

            top_n = st.slider(
                "Show top N numbers by total frequency",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
                key=f"top_n_freq_{group}",
            )

            show_only_seen = st.checkbox(
                "Hide numbers that never appeared",
                value=True,
                key=f"hide_never_{group}",
            )

            tbl = freq_tbl.copy()
            if show_only_seen:
                tbl = tbl[tbl["is_seen"]]

            tbl_sorted = tbl.sort_values(
                ["total_freq", "freq_90"], ascending=False
            ).head(top_n)

            st.dataframe(
                tbl_sorted[
                    [
                        "number",
                        "tag",
                        "total_freq",
                        "freq_90",
                        "freq_60",
                        "freq_30",
                        "last_seen_date",
                        "last_seen_days_ago",
                    ]
                ],
                use_container_width=True,
            )

            st.caption(
                "Tags: **Hot** = high recent frequency & seen very recently; "
                "**Warm** = decent recent frequency; **Cold** = not seen for a long time; "
                "**Never** = never observed in this group."
            )

        # =============================
        # Gap / cycle tab
        # =============================
        with gap_tab:
            st.markdown("### Gap / Cycle behaviour")

            gap_tbl = compute_gap_table(df_filtered, group)

            min_appearances = st.slider(
                "Minimum appearances to show",
                min_value=0,
                max_value=int(gap_tbl["appearances"].max()),
                value=min(10, int(gap_tbl["appearances"].max())),
                step=1,
                key=f"min_app_{group}",
            )

            gap_tbl_filtered = gap_tbl[gap_tbl["appearances"] >= min_appearances]

            st.dataframe(
                gap_tbl_filtered.sort_values(
                    ["appearances", "mean_gap_days"], ascending=[False, True]
                ),
                use_container_width=True,
            )

            selected_num = st.number_input(
                "Show raw gap distribution for number",
                min_value=0,
                max_value=99,
                value=0,
                step=1,
                key=f"gap_number_{group}",
            )

            gaps_series = compute_gaps_for_number(df_filtered, group, selected_num)
            if gaps_series is None or gaps_series.empty:
                st.info(
                    f"Number {selected_num} has fewer than 2 appearances in {group}, "
                    "so no gap distribution."
                )
            else:
                st.write(
                    f"Gap distribution for number **{selected_num}** in {group} "
                    f"(days between appearances):"
                )
                st.bar_chart(gaps_series.value_counts().sort_index())

        # =============================
        # Co-occurrence tab
        # =============================
        with co_tab:
            st.markdown("### Cross-group co-occurrence")

            other_groups = [g for g in GROUP_COLS if g != group]
            target_group = st.selectbox(
                "Target group",
                options=other_groups,
                index=0,
                key=f"target_{group}",
            )

            pair_stats = compute_pair_stats(df_filtered, group, target_group)

            base_num = st.number_input(
                f"Filter by base number in {group} (optional, -1 = show all)",
                min_value=-1,
                max_value=99,
                value=-1,
                step=1,
                key=f"base_filter_{group}",
            )

            if base_num >= 0:
                pair_stats = pair_stats[pair_stats["base_number"] == base_num]

            if pair_stats.empty:
                st.info("No valid co-occurrence data for these settings.")
            else:
                min_base_count = st.slider(
                    "Minimum base appearances",
                    min_value=1,
                    max_value=int(pair_stats["base_count"].max()),
                    value=min(5, int(pair_stats["base_count"].max())),
                    step=1,
                    key=f"min_base_count_{group}",
                )

                min_conf = st.slider(
                    "Minimum confidence (probability that target appears when base appears)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    key=f"min_conf_{group}",
                )

                min_lift = st.slider(
                    "Minimum lift (strength vs random chance)",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.1,
                    step=0.1,
                    key=f"min_lift_{group}",
                )

                filtered_pairs = pair_stats[
                    (pair_stats["base_count"] >= min_base_count)
                    & (pair_stats["confidence"] >= min_conf)
                    & (pair_stats["lift"] >= min_lift)
                ].copy()

                filtered_pairs = filtered_pairs.sort_values(
                    ["lift", "confidence", "pair_count"],
                    ascending=False,
                )

                st.dataframe(
                    filtered_pairs.head(100),
                    use_container_width=True,
                )

            st.caption(
                "💡 **Lift > 1** means the pair appears together more often than random chance. "
                "Higher lift & confidence = stronger co-occurrence pattern."
            )

        # =============================
        # Cluster view tab
        # =============================
        with cluster_tab:
            st.markdown("### Behavioural clusters (K-Means)")

            merged, feat = compute_cluster_features(df_filtered, group)

            max_clusters = min(8, len(feat))
            n_clusters = st.slider(
                "Number of clusters",
                min_value=1,
                max_value=max_clusters if max_clusters >= 1 else 1,
                value=min(4, max_clusters) if max_clusters >= 1 else 1,
                step=1,
                key=f"clusters_{group}",
            )

            merged_with_clusters = run_clustering(merged.copy(), feat, n_clusters)

            st.dataframe(
                merged_with_clusters[
                    [
                        "number",
                        "cluster",
                        "tag",
                        "total_freq",
                        "freq_90",
                        "mean_gap_days",
                        "std_gap_days",
                        "last_seen_days_ago",
                        "rhythm",
                    ]
                ].sort_values(["cluster", "total_freq"], ascending=[True, False]),
                use_container_width=True,
            )

            st.caption(
                "Each cluster groups numbers in this group that behave similarly "
                "in terms of frequency, recency, and gap structure."
            )

st.markdown("---")
st.markdown(
    "Built for exploring **patterns in DR / SG / FB / GZ / GL** "
    "without using future information or labels — pure pattern mining."
)
