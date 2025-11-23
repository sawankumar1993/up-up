import re
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="0–99 ML + Monthly Probability (CatBoost)",
    layout="wide",
)

st.title("0–99 ML + Monthly Probability Blending (Monthly-only mode)")
st.caption(
    "Step 1: Monthly-only probability strategy blended with a CatBoost model. "
    "Upload your lottery CSV (Year, Month, Day, DR, FB, GZ/GB, GL) to begin."
)

# -----------------------------
# Helpers
# -----------------------------
def find_col(df: pd.DataFrame, candidates):
    cand_lower = [c.lower() for c in candidates]
    for col in df.columns:
        if str(col).strip().lower() in cand_lower:
            return col
    return None


def parse_numbers(cell):
    if pd.isna(cell):
        return []
    return [int(x) for x in re.findall(r"\d{1,3}", str(cell))]


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    # pandas can sniff delimiter if sep=None, engine="python"
    df = pd.read_csv(file, sep=None, engine="python")
    return df


def prepare_base_df(df_raw: pd.DataFrame):
    # Find key columns (case-insensitive, supports Month/MM, Day/DD/DOM, GZ or GB)
    year_col = find_col(df_raw, ["year"])
    month_col = find_col(df_raw, ["month", "mm"])
    day_col = find_col(df_raw, ["day", "dd", "dom"])
    dr_col = find_col(df_raw, ["dr"])
    fb_col = find_col(df_raw, ["fb"])
    gz_col = find_col(df_raw, ["gz", "gb"])
    gl_col = find_col(df_raw, ["gl"])

    missing = [name for name, col in [
        ("Year", year_col),
        ("Month", month_col),
        ("Day", day_col),
        ("DR", dr_col),
        ("FB", fb_col),
        ("GZ/GB", gz_col),
        ("GL", gl_col),
    ] if col is None]

    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    # Build normalized date column
    df = df_raw.copy()
    df["__year"] = df[year_col].astype(int)
    df["__month"] = df[month_col].astype(int)
    df["__day"] = df[day_col].astype(int)

    df["date"] = pd.to_datetime(
        dict(year=df["__year"], month=df["__month"], day=df["__day"]),
        errors="coerce",
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Parsed numbers per lottery column
    df["DR_nums"] = df[dr_col].apply(parse_numbers)
    df["FB_nums"] = df[fb_col].apply(parse_numbers)
    df["GZGB_nums"] = df[gz_col].apply(parse_numbers)
    df["GL_nums"] = df[gl_col].apply(parse_numbers)

    return df


def build_number_day_dataset(df: pd.DataFrame, nums_max: int, which: str) -> pd.DataFrame:
    """
    Expand each date into rows for numbers 0..nums_max, with y=1 if that number
    appears in the chosen lottery column.
    which: one of "DR_nums", "FB_nums", "GZGB_nums", "GL_nums"
    """
    rows = []
    for _, row in df.iterrows():
        d = row["date"]
        month = d.month
        dow = d.weekday()  # Monday=0
        dom = d.day
        drawn_set = set(row[which])
        for n in range(nums_max + 1):
            rows.append(
                {
                    "date": d,
                    "month": month,
                    "dow": dow,
                    "dom": dom,
                    "num": n,
                    "y": 1 if n in drawn_set else 0,
                }
            )
    out = pd.DataFrame(rows)
    return out


def chronological_split(df_events: pd.DataFrame, train_frac: float = 0.8):
    """Time-based split by unique dates (leak-free)."""
    unique_dates = np.sort(df_events["date"].unique())
    if len(unique_dates) < 5:
        # too small, just use everything for train & valid
        idx = df_events.index
        return idx, idx, unique_dates[-1]

    split_idx = int(len(unique_dates) * train_frac)
    split_date = unique_dates[split_idx]

    train_idx = df_events.index[df_events["date"] < split_date]
    valid_idx = df_events.index[df_events["date"] >= split_date]
    # if split degenerates, ensure non-empty valid
    if len(valid_idx) == 0:
        valid_idx = train_idx
    return train_idx, valid_idx, split_date


@st.cache_resource(show_spinner=True)
def train_catboost_model(df_events: pd.DataFrame):
    """
    Train a CatBoost classifier on (month, dow, dom, num) -> y.
    Uses a chronological split to avoid leakage.
    """
    train_idx, valid_idx, split_date = chronological_split(df_events)
    features = ["month", "dow", "dom", "num"]

    X_train = df_events.loc[train_idx, features]
    y_train = df_events.loc[train_idx, "y"]
    X_valid = df_events.loc[valid_idx, features]
    y_valid = df_events.loc[valid_idx, "y"]

    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=6,
        learning_rate=0.1,
        iterations=300,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    # Simple validation metric
    valid_probs = model.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_probs >= 0.5).astype(int)
    acc = (valid_pred == y_valid.values).mean()

    return model, split_date, float(acc)


def compute_monthly_probs(
    df_base: pd.DataFrame,
    lottery_col: str,
    split_date,
    max_num: int,
    alpha: float,
):
    """
    Compute monthly P(number|month) using ONLY training window (date < split_date)
    to keep it leak-free.

    df_base: original per-date dataframe with *_nums lists
    lottery_col: e.g. "DR_nums"
    """
    # restrict to training dates
    df_train = df_base[df_base["date"] < split_date].copy()
    if df_train.empty:
        # fallback: use all data
        df_train = df_base.copy()

    monthly_totals = np.zeros(12, dtype=int)
    monthly_counts = np.zeros((12, max_num + 1), dtype=int)

    for _, row in df_train.iterrows():
        m = row["date"].month - 1  # 0-11
        monthly_totals[m] += 1
        for n in row[lottery_col]:
            if 0 <= n <= max_num:
                monthly_counts[m, n] += 1

    def probs_for_month(month: int):
        idx = month - 1
        tot = monthly_totals[idx]
        if tot == 0:
            # If no history for that month in training, fallback to uniform
            return np.full(max_num + 1, 1.0 / (max_num + 1))
        return (monthly_counts[idx, :] + alpha) / (tot + alpha * (max_num + 1))

    return probs_for_month, monthly_counts, monthly_totals


# -----------------------------
# Sidebar: data + settings
# -----------------------------
with st.sidebar:
    st.header("1. Data & Settings")
    file = st.file_uploader("Upload CSV", type=["csv", "tsv"])

    max_num = st.number_input(
        "Max number (inclusive)",
        min_value=10,
        max_value=999,
        value=99,
        step=1,
        help="If your lottery uses 0–99, keep at 99.",
    )

    blend_weight = st.slider(
        "ML / Probability blending (1.0 = pure Monthly probability)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="0.0 = pure CatBoost ML, 1.0 = pure monthly probability.",
    )

    alpha = st.number_input(
        "Laplace smoothing α (monthly probs)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.5,
    )

    top_n = st.number_input(
        "Top N numbers",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )

    core_n = st.number_input(
        "Core size",
        min_value=0,
        max_value=200,
        value=8,
        step=1,
        help="Highest confidence set. Must be ≤ Top N.",
    )

    mid_n = st.number_input(
        "Mid size",
        min_value=0,
        max_value=200,
        value=6,
        step=1,
        help="Medium confidence set. Core + Mid ≤ Top N; Edge = remaining.",
    )

    if core_n + mid_n > top_n:
        st.warning("Core + Mid must be ≤ Top N. Adjust the sizes.")

    run_button = st.button("Train model & score numbers", type="primary")

# -----------------------------
# Main logic
# -----------------------------
if not file:
    st.info("⬅️ Upload your CSV in the sidebar to get started.")
    st.stop()

# Load and normalize data
try:
    df_raw = load_csv(file)
    df_base = prepare_base_df(df_raw)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

st.success(
    f"Loaded {len(df_base)} draw days "
    f"from {df_base['date'].min().date()} to {df_base['date'].max().date()}."
)

# Choose lottery column
lottery_map = {
    "DR": "DR_nums",
    "FB": "FB_nums",
    "GZ/GB": "GZGB_nums",
    "GL": "GL_nums",
}
lottery_choice = st.selectbox(
    "Lottery column for modeling",
    options=list(lottery_map.keys()),
    index=0,
)
lottery_col = lottery_map[lottery_choice]

# Build per-number dataset
st.markdown("### Step 1 — Build per-number dataset")
with st.expander("Show details of expanded training data (optional)", expanded=False):
    st.write(
        "We expand each draw day into rows for every number (0…max). "
        "Each row is labeled 1 if the number appeared in the chosen lottery column, else 0."
    )

df_events = build_number_day_dataset(df_base, int(max_num), lottery_col)

st.write(
    f"Expanded to **{len(df_events):,} rows** "
    f"({len(df_events) // (int(max_num) + 1):,} days × {int(max_num) + 1} numbers)."
)

# Target prediction date
default_target = (df_base["date"].max() + pd.Timedelta(days=1)).date()
target_date = st.date_input(
    "Target draw date for prediction",
    min_value=df_base["date"].min().date(),
    max_value=(df_base["date"].max() + pd.Timedelta(days=365)).date(),
    value=default_target,
)

if not run_button:
    st.info("Set your parameters in the sidebar and click **Train model & score numbers**.")
    st.stop()

# -----------------------------
# Train CatBoost model
# -----------------------------
with st.spinner("Training CatBoost model (monthly-only features)…"):
    model, split_date, valid_acc = train_catboost_model(df_events)

st.success(
    f"CatBoost trained. Validation accuracy (time-based split @ {split_date.date()}): "
    f"**{valid_acc*100:.2f}%**"
)

# -----------------------------
# Monthly-only probability engine
# -----------------------------
probs_for_month_fn, monthly_counts, monthly_totals = compute_monthly_probs(
    df_base, lottery_col, split_date, int(max_num), alpha
)

target_month = target_date.month
month_probs = probs_for_month_fn(target_month)

# -----------------------------
# ML scores for target date
# -----------------------------
target_ts = pd.Timestamp(target_date)
target_features = pd.DataFrame(
    {
        "month": [target_ts.month] * (int(max_num) + 1),
        "dow": [target_ts.weekday()] * (int(max_num) + 1),
        "dom": [target_ts.day] * (int(max_num) + 1),
        "num": list(range(int(max_num) + 1)),
    }
)

ml_probs = model.predict_proba(target_features)[:, 1]

# -----------------------------
# Blend ML + monthly-only probability
# -----------------------------
final_scores = (1.0 - blend_weight) * ml_probs + blend_weight * month_probs

results = pd.DataFrame(
    {
        "number": np.arange(int(max_num) + 1),
        "ml_prob": ml_probs,
        "month_prob": month_probs,
        "final_score": final_scores,
    }
).sort_values("final_score", ascending=False)

st.markdown("### Step 2 — Ranked numbers (Monthly-only probability strategy)")

# Core / Mid / Edge
if core_n + mid_n > top_n:
    st.error("Core + Mid must be ≤ Top N. Please adjust sizes in the sidebar.")
else:
    top_df = results.head(int(top_n)).reset_index(drop=True)
    core_df = top_df.iloc[: int(core_n)]
    mid_df = top_df.iloc[int(core_n) : int(core_n + mid_n)]
    edge_df = top_df.iloc[int(core_n + mid_n) :]

    col_core, col_mid, col_edge = st.columns(3)

    with col_core:
        st.subheader(f"Core ({len(core_df)})")
        st.dataframe(
            core_df[["number", "final_score", "ml_prob", "month_prob"]],
            use_container_width=True,
        )

    with col_mid:
        st.subheader(f"Mid ({len(mid_df)})")
        st.dataframe(
            mid_df[["number", "final_score", "ml_prob", "month_prob"]],
            use_container_width=True,
        )

    with col_edge:
        st.subheader(f"Edge ({len(edge_df)})")
        st.dataframe(
            edge_df[["number", "final_score", "ml_prob", "month_prob"]],
            use_container_width=True,
        )

    st.markdown("#### Full ranking")
    st.dataframe(results, use_container_width=True)
