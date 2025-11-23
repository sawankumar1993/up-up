import io
import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

# Try importing CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ===================== Helper functions ===================== #

def detect_delimiter(first_line: str) -> str:
    """Detect delimiter similar to your JS detectDelimiter()."""
    if "\t" in first_line:
        return "\t"
    counts = {
        ";": first_line.count(";"),
        ",": first_line.count(","),
        "|": first_line.count("|"),
    }
    # Return delimiter with max count (fallback to comma)
    delim = max(counts, key=counts.get)
    return delim or ","


def month_index_full(name: str):
    """Map month name or prefix to 1–12 (like JS monthIndexFull)."""
    if name is None:
        return None
    m = str(name).strip().lower()
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    if m in months:
        return months[m]
    pref = m[:3]
    for full, idx in months.items():
        if full.startswith(pref):
            return idx
    return None


def numbers_from_cell(cell):
    """Extract 1–3 digit integers from a cell string."""
    s = "" if cell is None else str(cell)
    tokens = re.findall(r"\d{1,3}", s)
    nums = []
    for t in tokens:
        try:
            n = int(t)
            nums.append(n)
        except ValueError:
            pass
    return nums


def parse_custom_nums(text: str):
    """Parse custom number list (0–99) from text."""
    if not text:
        return set()
    tokens = re.findall(r"\d{1,3}", text)
    nums = set()
    for t in tokens:
        try:
            n = int(t)
            if 0 <= n <= 99:
                nums.add(n)
        except ValueError:
            continue
    return nums


def in_range_or_custom(n, rmin, rmax, custom_set, custom_only, use_custom) -> bool:
    """Replicates JS inRangeOrCustom()."""
    if custom_only and use_custom:
        return n in custom_set
    if use_custom and len(custom_set) > 0 and not custom_only:
        return (rmin <= n <= rmax) or (n in custom_set)
    return rmin <= n <= rmax


def load_csv_to_map(uploaded_file):
    """
    Parse CSV into a map keyed by yyyy-mm-dd:
    { key: {"DR": [...], "FB":[...], "GZGB":[...], "GL":[...]} }
    """
    raw_bytes = uploaded_file.getvalue()
    decoded = raw_bytes.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    if not lines:
        return {}, 0, True

    first_line = lines[0]
    delim = detect_delimiter(first_line)

    df = pd.read_csv(io.StringIO(decoded), delimiter=delim)
    df.columns = [str(c).strip().lower() for c in df.columns]

    col_year = "year" if "year" in df.columns else None
    col_month = None
    for cand in ["month", "mm"]:
        if cand in df.columns:
            col_month = cand
            break
    col_day = None
    for cand in ["day", "dd", "dom"]:
        if cand in df.columns:
            col_day = cand
            break

    if col_year is None or col_month is None or col_day is None:
        raise ValueError(
            "CSV must contain headers for year, month/mm and day/dd/dom (case-insensitive)."
        )

    col_dr = "dr" if "dr" in df.columns else None
    col_fb = "fb" if "fb" in df.columns else None
    col_gz = None
    has_gz_col = False
    if "gz" in df.columns:
        col_gz = "gz"
        has_gz_col = True
    elif "gb" in df.columns:
        col_gz = "gb"
        has_gz_col = False

    col_gl = "gl" if "gl" in df.columns else None

    date_map = {}
    seen_keys = set()

    for _, row in df.iterrows():
        y_raw = str(row[col_year]).strip() if col_year in row else ""
        m_raw = str(row[col_month]).strip() if col_month in row else ""
        d_raw = str(row[col_day]).strip() if col_day in row else ""
        if not y_raw or not d_raw:
            continue
        try:
            y = int(float(y_raw))
            d = int(float(d_raw))
        except ValueError:
            continue
        if y <= 0 or d <= 0:
            continue

        # month can be number or name
        if m_raw.isdigit():
            m = int(float(m_raw))
        else:
            m = month_index_full(m_raw)
        if not m or m < 1 or m > 12:
            continue

        key = f"{y:04d}-{m:02d}-{d:02d}"
        entry = date_map.get(key, {"DR": [], "FB": [], "GZGB": [], "GL": []})

        if col_dr is not None:
            entry["DR"].extend(numbers_from_cell(row[col_dr]))
        if col_fb is not None:
            entry["FB"].extend(numbers_from_cell(row[col_fb]))
        if col_gz is not None:
            entry["GZGB"].extend(numbers_from_cell(row[col_gz]))
        if col_gl is not None:
            entry["GL"].extend(numbers_from_cell(row[col_gl]))

        date_map[key] = entry
        seen_keys.add(key)

    return date_map, len(seen_keys), has_gz_col


def build_rows_df(date_map, rmin, rmax, custom_set, custom_only):
    """Replicate JS buildRows(), returning a pandas DataFrame."""
    keys = sorted(date_map.keys())
    rows = []
    use_custom = len(custom_set) > 0

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for key in keys:
        entry = date_map.get(key, {"DR": [], "FB": [], "GZGB": [], "GL": []})
        dr_list = entry.get("DR", []) or []
        fb_list = entry.get("FB", []) or []
        gz_list = entry.get("GZGB", []) or []
        gl_list = entry.get("GL", []) or []

        def check_arr(arr):
            for n in arr:
                if in_range_or_custom(n, rmin, rmax, custom_set, custom_only, use_custom):
                    return True
            return False

        dr_win = check_arr(dr_list)
        fb_win = check_arr(fb_list)
        gz_win = check_arr(gz_list)
        gl_win = check_arr(gl_list)
        any_win = dr_win or fb_win or gz_win or gl_win

        # parse date
        y, m, d = [int(x) for x in key.split("-")]
        dt = datetime(y, m, d)
        dow_idx = dt.weekday()  # 0=Mon
        dow_label = dow_labels[dow_idx]

        rows.append(
            {
                "date_key": key,
                "date": dt,
                "dow": dow_label,
                "dow_idx": dow_idx,
                "DR": dr_list,
                "FB": fb_list,
                "GZGB": gz_list,
                "GL": gl_list,
                "DR_win": dr_win,
                "FB_win": fb_win,
                "GZGB_win": gz_win,
                "GL_win": gl_win,
                "any_win": any_win,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9  # seconds
    return df


def summarize_wins(df: pd.DataFrame):
    total_days = len(df)
    return {
        "total_days": total_days,
        "DR": int(df["DR_win"].sum()),
        "FB": int(df["FB_win"].sum()),
        "GZGB": int(df["GZGB_win"].sum()),
        "GL": int(df["GL_win"].sum()),
        "ANY": int(df["any_win"].sum()),
    }


def format_nums(arr):
    if not isinstance(arr, (list, tuple)):
        return ""
    return " ".join(str(x) for x in sorted(arr))


def analyze_prob(df: pd.DataFrame, mode: str = "dow", alpha: float = 2.0,
                 lam: float = 0.01, advanced: bool = True):
    """Python version of your JS analyzeProb()."""
    if df.empty:
        return None

    now_ts = df["timestamp"].max()
    if mode == "dow":
        buckets = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        bucket_series = df["dow"]
    else:
        buckets = [str(i) for i in range(1, 32)]
        bucket_series = df["date"].dt.day.astype(str)

    K = len(buckets)
    counts = {b: 0 for b in buckets}
    totals = {b: 0 for b in buckets}
    weighted_hits = {b: 0.0 for b in buckets}
    weighted_totals = {b: 0.0 for b in buckets}

    for i, row in df.iterrows():
        label = bucket_series.iloc[i]
        if label not in totals:
            continue
        totals[label] += 1
        is_hit = bool(row["any_win"])
        if is_hit:
            counts[label] += 1

        age_days = max(0.0, (now_ts - row["timestamp"]) / 86400.0)
        w = float(np.exp(-lam * age_days))
        weighted_totals[label] += w
        if is_hit:
            weighted_hits[label] += w

    results = []
    for b in buckets:
        hit = counts[b]
        tot = totals[b]
        basic = (hit / tot) if tot > 0 else 0.0
        smoothed = (hit + alpha) / (tot + alpha * K) if (tot + alpha * K) > 0 else 0.0
        weighted = (weighted_hits[b] / weighted_totals[b]) if weighted_totals[b] > 0 else 0.0
        final = (0.6 * weighted + 0.3 * smoothed + 0.1 * basic) if advanced else basic
        results.append(
            {
                "bucket": b,
                "hit": hit,
                "total": tot,
                "basic": basic,
                "smoothed": smoothed,
                "weighted": weighted,
                "final": final,
            }
        )

    if mode == "dom":
        results.sort(key=lambda r: int(r["bucket"]))

    return {
        "results": results,
        "meta": {
            "alpha": alpha,
            "lambda": lam,
            "advanced": advanced,
            "K": K,
        },
    }


def make_ml_features(df: pd.DataFrame, rmin: int, rmax: int, custom_set: set):
    """Features for CatBoost model; one row per date."""
    feats = pd.DataFrame()
    feats["dow_idx"] = df["dow_idx"]
    feats["dom"] = df["date"].dt.day
    feats["month"] = df["date"].dt.month
    feats["year"] = df["date"].dt.year
    feats["range_span"] = rmax - rmin
    feats["num_custom"] = len(custom_set)
    # Optional: counts of numbers in each column (static per date)
    feats["len_DR"] = df["DR"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_FB"] = df["FB"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_GZGB"] = df["GZGB"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_GL"] = df["GL"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    return feats


def chronological_train_test_split(df_features, y, train_ratio=0.8):
    """Time-series safe split: first N% train, last (1-N%) test."""
    sort_idx = np.argsort(df_features.index.values)  # already chronological if df sorted
    X_sorted = df_features.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)

    n = len(X_sorted)
    n_train = max(1, int(n * train_ratio))
    if n_train >= n:
        n_train = n - 1  # keep at least one test point
    X_train = X_sorted.iloc[:n_train]
    y_train = y_sorted.iloc[:n_train]
    X_test = X_sorted.iloc[n_train:]
    y_test = y_sorted.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def train_catboost(df: pd.DataFrame, rmin: int, rmax: int, custom_set: set, train_ratio: float = 0.8):
    feats = make_ml_features(df, rmin, rmax, custom_set)
    y = df["any_win"].astype(int)
    X_train, X_test, y_train, y_test = chronological_train_test_split(feats, y, train_ratio=train_ratio)

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        iterations=400,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    proba_test = model.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)

    metrics = {}
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["accuracy"] = float(accuracy_score(y_test, preds_test))
    metrics["precision"] = float(precision_score(y_test, preds_test, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, preds_test, zero_division=0))
    try:
        metrics["auc"] = float(roc_auc_score(y_test, proba_test))
    except ValueError:
        metrics["auc"] = float("nan")

    return model, metrics


def make_single_features_for_date(target_date: date, rmin: int, rmax: int, custom_set: set):
    """Create one-row feature DataFrame for CatBoost prediction."""
    dt = datetime(target_date.year, target_date.month, target_date.day)
    df_tmp = pd.DataFrame({"date": [dt], "dow_idx": [dt.weekday()]})
    df_tmp["DR"] = [[]]
    df_tmp["FB"] = [[]]
    df_tmp["GZGB"] = [[]]
    df_tmp["GL"] = [[]]
    return make_ml_features(df_tmp, rmin, rmax, custom_set)


# ===================== Streamlit App ===================== #

st.set_page_config(
    page_title="Lottery Range Wins — Probability + CatBoost",
    layout="wide",
)

st.title("Lottery Range Wins — Probability + CatBoost (Streamlit)")

st.markdown(
    """
Upload the same CSV you use in the browser app (Year, Month, Day, DR, FB, GZ/GB, GL),
choose your range/custom numbers, and this app will:

1. Show WIN status per day and a summary  
2. Compute **Bayesian + recency-weighted probabilities** (Day-of-Week or Day-of-Month)  
3. Train a **CatBoost model** (time-series safe) to predict the chance your range/custom will WIN on a date  
"""
)

# --- Input area --- #
col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded = st.file_uploader("Upload CSV", type=["csv", "tsv"])

with col_right:
    rmin = st.number_input("Range min (0–99)", min_value=0, max_value=99, value=0)
    rmax = st.number_input("Range max (0–99)", min_value=0, max_value=99, value=9)
    custom_text = st.text_input("Custom numbers (comma/space separated)", placeholder="e.g. 5, 11, 44, 88")
    custom_only = st.checkbox("Use only custom numbers (ignore range)")

if rmin > rmax:
    st.warning("Range min is greater than max — values will be swapped internally.")
    rmin, rmax = rmax, rmin

custom_set = parse_custom_nums(custom_text)
use_custom = len(custom_set) > 0

analyze_clicked = st.button("Analyze")

if analyze_clicked and not uploaded:
    st.error("Please upload a CSV first.")

if "built_df" not in st.session_state:
    st.session_state["built_df"] = None
    st.session_state["has_gz"] = True
    st.session_state["range"] = (rmin, rmax)
    st.session_state["custom_set"] = custom_set
    st.session_state["custom_only"] = custom_only
    st.session_state["cb_model"] = None
    st.session_state["cb_metrics"] = None

# --- Main logic after Analyze --- #
if uploaded and analyze_clicked:
    try:
        date_map, count_dates, has_gz = load_csv_to_map(uploaded)
        built_df = build_rows_df(date_map, rmin, rmax, custom_set, custom_only)
        st.session_state["built_df"] = built_df
        st.session_state["has_gz"] = has_gz
        st.session_state["range"] = (rmin, rmax)
        st.session_state["custom_set"] = custom_set
        st.session_state["custom_only"] = custom_only
        st.session_state["cb_model"] = None
        st.session_state["cb_metrics"] = None

        st.success(f"Parsed {count_dates} unique dates from CSV.")
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")

built_df = st.session_state.get("built_df", None)

if built_df is not None and not built_df.empty:
    rmin, rmax = st.session_state["range"]
    custom_set = st.session_state["custom_set"]
    custom_only = st.session_state["custom_only"]
    has_gz = st.session_state["has_gz"]

    # ----- Summary ----- #
    st.subheader("Summary")

    summary = summarize_wins(built_df)
    s_cols = st.columns(5)
    s_cols[0].metric("Total days", summary["total_days"])
    s_cols[1].metric("DR WIN days", summary["DR"])
    s_cols[2].metric("FB WIN days", summary["FB"])
    s_cols[3].metric("GZ/GB WIN days", summary["GZGB"])
    s_cols[4].metric("GL WIN days", summary["GL"])

    st.caption(f"Any WIN (days): **{summary['ANY']}**  • Range: `{rmin}-{rmax}`  • Custom: {sorted(list(custom_set)) or 'None'}")

    # ----- Table ----- #
    st.subheader("Per-day Table")

    table_df = pd.DataFrame(
        {
            "#": range(1, len(built_df) + 1),
            "Date": built_df["date_key"],
            "Day": built_df["dow"],
            "DR": built_df["DR"].apply(format_nums),
            "DR Status": np.where(built_df["DR_win"], "WIN", "NOT WIN"),
            "FB": built_df["FB"].apply(format_nums),
            "FB Status": np.where(built_df["FB_win"], "WIN", "NOT WIN"),
            "GZ/GB": built_df["GZGB"].apply(format_nums),
            "GZ/GB Status": np.where(built_df["GZGB_win"], "WIN", "NOT WIN"),
            "GL": built_df["GL"].apply(format_nums),
            "GL Status": np.where(built_df["GL_win"], "WIN", "NOT WIN"),
            "Any WIN": np.where(built_df["any_win"], "WIN", "NOT WIN"),
        }
    )

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Download CSV of built table
    download_df = table_df.copy()
    download_df["Range"] = f"{rmin}-{rmax}"
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analyzed CSV",
        data=csv_bytes,
        file_name="lottery_range_wins.csv",
        mime="text/csv",
    )

    # ----- Probability Engine ----- #
    st.subheader("Probability Engine (Bayesian + Recency Weighted)")

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Mode", ["Day-of-Week", "Day-of-Month"])
    alpha = c2.number_input("Alpha (smoothing)", min_value=0.0, value=2.0, step=0.5)
    lam = c3.number_input("Decay λ (recency)", min_value=0.0, value=0.01, step=0.005, format="%0.5f")
    advanced = c4.checkbox("Use advanced blend", value=True)

    if st.button("Run Probability"):
        prob_mode = "dow" if mode == "Day-of-Week" else "dom"
        prob_res = analyze_prob(built_df, mode=prob_mode, alpha=alpha, lam=lam, advanced=advanced)
        if prob_res is None:
            st.warning("No data.")
        else:
            prob_rows = prob_res["results"]
            prob_df = pd.DataFrame(prob_rows)
            prob_df_display = prob_df.copy()
            for col in ["basic", "smoothed", "weighted", "final"]:
                prob_df_display[col] = (prob_df_display[col] * 100.0).round(2)

            st.write("**Probability Table (%):**")
            st.dataframe(prob_df_display, use_container_width=True, hide_index=True)

            st.write("**Final probability per bucket (bar):**")
            chart_df = prob_df[["bucket", "final"]].set_index("bucket")
            st.bar_chart(chart_df)

            st.write("**Basic / Smoothed / Weighted curves:**")
            line_df = prob_df[["bucket", "basic", "smoothed", "weighted"]].set_index("bucket")
            st.line_chart(line_df)

    # ----- CatBoost ML: Predict WIN for range/custom ----- #
    st.subheader("CatBoost ML — Predict WIN probability for this Range/Custom")

    if not CATBOOST_AVAILABLE:
        st.error(
            "CatBoost is not installed in this environment.\n\n"
            "Install it with:\n`pip install catboost`"
        )
    else:
        train_ratio = st.slider("Train/Test split (chronological train size)",
                                min_value=0.6, max_value=0.9, value=0.8, step=0.05)

        if st.button("Train CatBoost model (time-series safe)"):
            with st.spinner("Training CatBoost model..."):
                model, metrics = train_catboost(built_df, rmin, rmax, custom_set, train_ratio=train_ratio)
                st.session_state["cb_model"] = model
                st.session_state["cb_metrics"] = metrics

            m = metrics
            m_cols = st.columns(5)
            m_cols[0].metric("Train samples", m["n_train"])
            m_cols[1].metric("Test samples", m["n_test"])
            m_cols[2].metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            m_cols[3].metric("Precision", f"{m['precision']*100:.2f}%")
            m_cols[4].metric("Recall", f"{m['recall']*100:.2f}%")
            st.caption(f"AUC: {m['auc']:.4f}")

        cb_model = st.session_state.get("cb_model", None)
        cb_metrics = st.session_state.get("cb_metrics", None)

        if cb_model is not None:
            st.success("CatBoost model is trained and ready.")

            pred_date = st.date_input("Pick a date to predict WIN probability", value=date.today())
            if st.button("Predict WIN probability for this date"):
                X_new = make_single_features_for_date(pred_date, rmin, rmax, custom_set)
                proba = cb_model.predict_proba(X_new)[0, 1]
                st.metric("Predicted WIN probability", f"{proba*100:.2f}%")
        else:
            st.info("Train the CatBoost model first to enable date-wise predictions.")

else:
    st.info("Upload a CSV and click **Analyze** to start.")
