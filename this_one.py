import io
import re
import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# -------------------- Constants -------------------- #

DOWS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


# -------------------- CSV + base parsing -------------------- #

def detect_delimiter(first_line: str) -> str:
    if "\t" in first_line:
        return "\t"
    counts = {
        ";": first_line.count(";"),
        ",": first_line.count(","),
        "|": first_line.count("|"),
    }
    return max(counts, key=counts.get) or ","


def numbers_from_cell(cell):
    s = "" if cell is None else str(cell)
    tokens = re.findall(r"\d{1,3}", s)
    out = []
    for t in tokens:
        try:
            n = int(t)
            out.append(n)
        except ValueError:
            pass
    return out


def month_index_full(name: str):
    if name is None:
        return None
    m = str(name).strip().lower()
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    if m in months:
        return months[m]
    pref = m[:3]
    for full, idx in months.items():
        if full.startswith(pref):
            return idx
    return None


def load_csv_to_map(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    decoded = raw_bytes.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    if not lines:
        return {}, 0, True

    delim = detect_delimiter(lines[0])
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
    has_gz = False
    if "gz" in df.columns:
        col_gz = "gz"
        has_gz = True
    elif "gb" in df.columns:
        col_gz = "gb"
        has_gz = False
    col_gl = "gl" if "gl" in df.columns else None

    date_map = {}
    seen_keys = set()

    for _, row in df.iterrows():
        y_raw = str(row[col_year]).strip()
        m_raw = str(row[col_month]).strip()
        d_raw = str(row[col_day]).strip()
        if not y_raw or not d_raw:
            continue
        try:
            y = int(float(y_raw))
            d = int(float(d_raw))
        except ValueError:
            continue
        if y <= 0 or d <= 0:
            continue

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

    return date_map, len(seen_keys), has_gz


def parse_custom_nums(text: str):
    if not text:
        return set()
    tokens = re.findall(r"\d{1,3}", text)
    out = set()
    for t in tokens:
        try:
            n = int(t)
            if 0 <= n <= 99:
                out.add(n)
        except ValueError:
            continue
    return out


def in_range_or_custom(n, rmin, rmax, custom_set, custom_only, use_custom) -> bool:
    if custom_only and use_custom:
        return n in custom_set
    if use_custom and custom_set and not custom_only:
        return (rmin <= n <= rmax) or (n in custom_set)
    return rmin <= n <= rmax


def build_rows_df(date_map, rmin, rmax, custom_set, custom_only):
    keys = sorted(date_map.keys())
    rows = []
    use_custom = bool(custom_set)

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

        y, m, d = [int(x) for x in key.split("-")]
        dt = datetime(y, m, d)

        # JS-style DOW index: 0=Sun, 6=Sat
        dow_js_idx = (dt.weekday() + 1) % 7  # Python weekday: 0=Mon
        dow_label = DOWS[dow_js_idx]

        rows.append(
            {
                "date_key": key,
                "date": dt,
                "dow_label": dow_label,
                "dow_js_idx": dow_js_idx,
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
        df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
    return df


def format_nums(arr):
    if not isinstance(arr, (list, tuple)):
        return ""
    return " ".join(str(int(x)) for x in sorted(arr))


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


# -------------------- Probability engine (per-day range) -------------------- #

def analyze_prob(df: pd.DataFrame, mode: str = "dow", alpha: float = 2.0,
                 lam: float = 0.01, advanced: bool = True):
    if df.empty:
        return None

    now_ts = df["timestamp"].max()

    if mode == "dow":
        buckets = DOWS
        bucket_series = df["dow_label"]
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

    return {"results": results}


# -------------------- ML — per-date range model (Any WIN) -------------------- #

def make_ml_features_range(df: pd.DataFrame, rmin: int, rmax: int, custom_set: set):
    feats = pd.DataFrame()
    feats["dow_js_idx"] = df["dow_js_idx"]
    feats["dom"] = df["date"].dt.day
    feats["month"] = df["date"].dt.month
    feats["year"] = df["date"].dt.year
    feats["range_span"] = rmax - rmin
    feats["num_custom"] = len(custom_set)
    feats["len_DR"] = df["DR"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_FB"] = df["FB"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_GZGB"] = df["GZGB"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    feats["len_GL"] = df["GL"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    return feats


def chrono_split_dates(df_features, y, train_ratio=0.8):
    sort_idx = np.argsort(df_features.index.values)
    X_sorted = df_features.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)

    n = len(X_sorted)
    n_train = max(1, int(n * train_ratio))
    if n_train >= n:
        n_train = n - 1
    X_train = X_sorted.iloc[:n_train]
    y_train = y_sorted.iloc[:n_train]
    X_test = X_sorted.iloc[n_train:]
    y_test = y_sorted.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def train_catboost_range(df: pd.DataFrame, rmin: int, rmax: int, custom_set: set, train_ratio: float = 0.8):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    feats = make_ml_features_range(df, rmin, rmax, custom_set)
    y = df["any_win"].astype(int)
    X_train, X_test, y_train, y_test = chrono_split_dates(feats, y, train_ratio=train_ratio)

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

    proba_test = model.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds_test)) if len(X_test) > 0 else float("nan"),
        "precision": float(precision_score(y_test, preds_test, zero_division=0)) if len(X_test) > 0 else float("nan"),
        "recall": float(recall_score(y_test, preds_test, zero_division=0)) if len(X_test) > 0 else float("nan"),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_test, proba_test)) if len(X_test) > 0 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")

    return model, metrics


def make_single_features_for_date_range(target_date: date, rmin: int, rmax: int, custom_set: set):
    dt = datetime(target_date.year, target_date.month, target_date.day)
    df_tmp = pd.DataFrame({"date": [dt]})
    dow_js_idx = (dt.weekday() + 1) % 7
    df_tmp["dow_js_idx"] = [dow_js_idx]
    df_tmp["DR"] = [[]]
    df_tmp["FB"] = [[]]
    df_tmp["GZGB"] = [[]]
    df_tmp["GL"] = [[]]
    feats = make_ml_features_range(df_tmp, rmin, rmax, custom_set)
    return feats


# -------------------- ML Ensemble 0–99 (CatBoost replacement) -------------------- #

def ens_build_hit_dates(df: pd.DataFrame, lottery_key: str):
    """Build per-number hit timestamp lists (ms since epoch), sorted by time."""
    hit_dates = [[] for _ in range(100)]
    df_sorted = df.sort_values("date")
    for _, row in df_sorted.iterrows():
        ts = int(row["timestamp"] * 1000)
        nums_arr = row[lottery_key] if isinstance(row[lottery_key], list) else []
        for raw in nums_arr:
            try:
                num = int(raw)
            except ValueError:
                continue
            if 0 <= num <= 99:
                hit_dates[num].append(ts)
    return hit_dates


def ens_build_dataset(
    df: pd.DataFrame,
    lottery_key: str,
    recency_half_life: float = 30.0,
    recency_window_days: float = 90.0,
):
    """Build leak-free per-(date, number) training dataset with recency features."""
    xs = []
    ys = []

    df_sorted = df.sort_values("date").reset_index(drop=True)
    hit_dates = ens_build_hit_dates(df_sorted, lottery_key)

    day_ms = 24 * 60 * 60 * 1000
    max_days_norm = max(recency_window_days * 2.0, 365.0)
    if recency_half_life > 0:
        gamma = math.log(2.0) / float(recency_half_life)
    else:
        gamma = 0.0

    for _, row in df_sorted.iterrows():
        key = row["date_key"]
        parts = str(key).split("-")
        if len(parts) != 3:
            continue
        try:
            dd = int(parts[2])
        except ValueError:
            continue

        dow_idx = int(row["dow_js_idx"])
        nums_arr = row[lottery_key] if isinstance(row[lottery_key], list) else []
        num_set = set(int(n) for n in nums_arr if 0 <= int(n) <= 99)

        today_ts = int(row["timestamp"] * 1000)

        x_base1 = dow_idx / 6.0
        x_base2 = dd / 31.0

        for num in range(100):
            ts_list = hit_dates[num]
            last_age_days = None
            count_window = 0

            if ts_list:
                for ts in reversed(ts_list):
                    if ts >= today_ts:
                        continue  # skip same-day/future hits
                    age_days = (today_ts - ts) / day_ms
                    if last_age_days is None:
                        last_age_days = age_days
                    if age_days <= recency_window_days:
                        count_window += 1
                    else:
                        break

            if last_age_days is None:
                recency_raw = 0.0
                days_norm = 1.0  # "very far in the past / never"
            else:
                recency_raw = math.exp(-gamma * last_age_days) if gamma > 0 else 0.0
                days_norm = min(last_age_days, max_days_norm) / max_days_norm

            freq_recent = count_window / recency_window_days

            x3 = num / 99.0
            x = [
                x_base1,
                x_base2,
                x3,
                recency_raw,
                freq_recent,
                days_norm,
            ]
            y_val = 1 if num in num_set else 0
            xs.append(x)
            ys.append(y_val)

    if not xs:
        return None, None

    X = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.int32)
    return X, y


def ens_train_catboost(
    df: pd.DataFrame,
    lottery_key: str,
    train_ratio: float = 0.8,
    recency_half_life: float = 30.0,
    recency_window_days: float = 90.0,
):
    """Train CatBoost on per-(date, num) dataset with class weighting for hits."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    X, y = ens_build_dataset(
        df,
        lottery_key,
        recency_half_life=recency_half_life,
        recency_window_days=recency_window_days,
    )
    if X is None:
        raise ValueError("No samples built for ensemble model.")

    n = X.shape[0]
    n_train = max(1, int(n * train_ratio))
    if n_train >= n:
        n_train = n - 1
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if pos_count > 0 and neg_count > 0:
        scale_pos_weight = float(neg_count) / float(pos_count)
    else:
        scale_pos_weight = 1.0

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.08,
        iterations=400,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    if len(X_val) > 0:
        proba_val = model.predict_proba(X_val)[:, 1]
        preds_val = (proba_val >= 0.5).astype(int)
        val_acc = float(accuracy_score(y_val, preds_val))
        try:
            val_auc = float(roc_auc_score(y_val, proba_val))
        except ValueError:
            val_auc = float("nan")
    else:
        val_acc = float("nan")
        val_auc = float("nan")

    metrics = {
        "samples": int(n),
        "n_train": int(n_train),
        "n_val": int(n - n_train),
        "val_accuracy": val_acc,
        "val_auc": val_auc,
        "pos_frac_train": float(pos_count) / float(len(y_train)) if len(y_train) > 0 else float("nan"),
        "scale_pos_weight": scale_pos_weight,
    }
    return model, metrics


def ens_auto_search_best_model(
    df: pd.DataFrame,
    lottery_key: str,
    train_ratio_list=None,
    half_life_list=None,
    window_list=None,
):
    """
    Hyperparameter search over:
    - train_ratio
    - recency_half_life
    - recency_window_days
    """
    if train_ratio_list is None:
        train_ratio_list = [0.7, 0.8, 0.85]
    if half_life_list is None:
        half_life_list = [15.0, 30.0, 60.0]
    if window_list is None:
        window_list = [60.0, 90.0, 120.0]

    history = []
    best_model = None
    best_config = None
    best_score = -1.0

    total_runs = len(train_ratio_list) * len(half_life_list) * len(window_list)
    run_idx = 0

    for tr in train_ratio_list:
        for hl in half_life_list:
            for wd in window_list:
                run_idx += 1
                st.write(f"Search run {run_idx}/{total_runs} — train_ratio={tr}, half_life={hl}, window={wd}")
                try:
                    model, metrics = ens_train_catboost(
                        df,
                        lottery_key=lottery_key,
                        train_ratio=tr,
                        recency_half_life=hl,
                        recency_window_days=wd,
                    )
                except Exception as e:
                    st.write(f"  ❌ Failed: {e}")
                    continue

                val_acc = metrics.get("val_accuracy", float("nan"))
                if np.isnan(val_acc):
                    score = -1.0
                else:
                    score = float(val_acc)

                row = {
                    "train_ratio": tr,
                    "recency_half_life": hl,
                    "recency_window_days": wd,
                    "val_accuracy": val_acc,
                    "val_auc": metrics.get("val_auc", float("nan")),
                    "samples": metrics.get("samples", np.nan),
                    "n_train": metrics.get("n_train", np.nan),
                    "n_val": metrics.get("n_val", np.nan),
                    "pos_frac_train": metrics.get("pos_frac_train", np.nan),
                    "scale_pos_weight": metrics.get("scale_pos_weight", np.nan),
                }
                history.append(row)

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_config = {
                        "train_ratio": tr,
                        "recency_half_life": hl,
                        "recency_window_days": wd,
                        "val_accuracy": val_acc,
                        "val_auc": metrics.get("val_auc", float("nan")),
                        "samples": metrics.get("samples", np.nan),
                        "n_train": metrics.get("n_train", np.nan),
                        "n_val": metrics.get("n_val", np.nan),
                        "pos_frac_train": metrics.get("pos_frac_train", np.nan),
                        "scale_pos_weight": metrics.get("scale_pos_weight", np.nan),
                        "source": "auto-search",
                    }

    history_df = pd.DataFrame(history) if history else pd.DataFrame()
    return best_model, best_config, history_df


def ens_build_number_profiles(df: pd.DataFrame, lottery_key: str, recency_mode: bool):
    """
    Weekly / monthly probability profiles.
    recency_mode=True -> exponential decay by age; False -> uniform.
    """
    now_ts = int(datetime.utcnow().timestamp() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    lam = 0.02

    hits_num_dow = np.zeros((100, 7), dtype=float)
    hits_num_dom = np.zeros((100, 32), dtype=float)
    tot_dow = np.zeros(7, dtype=float)
    tot_dom = np.zeros(32, dtype=float)

    df_sorted = df.sort_values("date")
    for _, row in df_sorted.iterrows():
        key = row["date_key"]
        parts = str(key).split("-")
        if len(parts) != 3:
            continue
        try:
            dd = int(parts[2])
        except ValueError:
            continue
        if dd < 1 or dd > 31:
            continue

        try:
            dow_index = DOWS.index(row["dow_label"])
        except ValueError:
            continue

        ts = int(row["timestamp"] * 1000)
        if recency_mode:
            age_days = max(0.0, (now_ts - ts) / day_ms)
            w = float(np.exp(-lam * age_days))
        else:
            w = 1.0

        tot_dow[dow_index] += w
        tot_dom[dd] += w

        nums_arr = row[lottery_key] if isinstance(row[lottery_key], list) else []
        for raw in nums_arr:
            try:
                num = int(raw)
            except ValueError:
                continue
            if 0 <= num <= 99:
                hits_num_dow[num, dow_index] += w
                hits_num_dom[num, dd] += w

    weekly_profile = np.zeros((100, 7), dtype=float)
    monthly_profile = np.zeros((100, 32), dtype=float)
    max_weekly = 0.0
    max_monthly = 0.0

    for num in range(100):
        for d in range(7):
            tot = tot_dow[d]
            if tot > 0:
                p = hits_num_dow[num, d] / tot
                weekly_profile[num, d] = p
                if p > max_weekly:
                    max_weekly = p
        for dom in range(1, 32):
            tot = tot_dom[dom]
            if tot > 0:
                p = hits_num_dom[num, dom] / tot
                monthly_profile[num, dom] = p
                if p > max_monthly:
                    max_monthly = p

    return {
        "weekly_profile": weekly_profile,
        "monthly_profile": monthly_profile,
        "max_weekly": max_weekly,
        "max_monthly": max_monthly,
    }


def ens_score_numbers_for_date(
    model,
    df: pd.DataFrame,
    lottery_key: str,
    prob_mode: str,
    prob_weight_coeff: float,
    prediction_date: date,
    recency_half_life: float = 30.0,
    recency_window_days: float = 90.0,
    recency_alpha: float = 0.7,
):
    """
    Score 0–99 for a given prediction date using:
    - CatBoost ensemble (ML)
    - Weekly / Monthly / Recency probability profiles (leak-free)
    """
    if df.empty:
        raise ValueError("No data loaded.")

    use_recency_profiles = prob_mode in ("weekly", "monthly", "both")
    profiles = ens_build_number_profiles(df, lottery_key, recency_mode=use_recency_profiles)
    weekly_profile = profiles["weekly_profile"]
    monthly_profile = profiles["monthly_profile"]
    max_weekly = profiles["max_weekly"]
    max_monthly = profiles["max_monthly"]

    hit_dates = ens_build_hit_dates(df, lottery_key)

    dt = datetime(prediction_date.year, prediction_date.month, prediction_date.day)
    dow_index = (dt.weekday() + 1) % 7  # JS-style 0..6, Sun=0
    dd = dt.day

    x1 = dow_index / 6.0
    x2 = dd / 31.0
    feats = []
    for num in range(100):
        x3 = num / 99.0

        day_ms = 24 * 60 * 60 * 1000
        today_ts = int(dt.timestamp() * 1000)
        if recency_half_life > 0:
            gamma = math.log(2.0) / float(recency_half_life)
        else:
            gamma = 0.0
        max_days_norm = max(recency_window_days * 2.0, 365.0)

        ts_list = hit_dates[num]
        last_age_days = None
        count_window = 0
        if ts_list:
            for ts in reversed(ts_list):
                if ts >= today_ts:
                    continue
                age_days = (today_ts - ts) / day_ms
                if last_age_days is None:
                    last_age_days = age_days
                if age_days <= recency_window_days:
                    count_window += 1
                else:
                    break

        if last_age_days is None:
            recency_raw = 0.0
            days_norm = 1.0
        else:
            recency_raw = math.exp(-gamma * last_age_days) if gamma > 0 else 0.0
            days_norm = min(last_age_days, max_days_norm) / max_days_norm

        freq_recent = count_window / recency_window_days

        feats.append([x1, x2, x3, recency_raw, freq_recent, days_norm])

    X_pred = np.array(feats, dtype=np.float32)
    ml_probs = model.predict_proba(X_pred)[:, 1]

    if prob_mode == "recent":
        day_ms = 24 * 60 * 60 * 1000
        today_ts = int(dt.timestamp() * 1000)
        if recency_half_life > 0:
            gamma = math.log(2.0) / float(recency_half_life)
        else:
            gamma = 0.0

        rec_raw_all = np.zeros(100, dtype=float)
        freq_all = np.zeros(100, dtype=float)

        for num in range(100):
            ts_list = hit_dates[num]
            last_age_days = None
            count_window = 0
            if ts_list:
                for ts in reversed(ts_list):
                    if ts >= today_ts:
                        continue
                    age_days = (today_ts - ts) / day_ms
                    if last_age_days is None:
                        last_age_days = age_days
                    if age_days <= recency_window_days:
                        count_window += 1
                    else:
                        break

            if last_age_days is not None and gamma > 0:
                rec_raw_all[num] = math.exp(-gamma * last_age_days)
            else:
                rec_raw_all[num] = 0.0

            freq_all[num] = count_window / recency_window_days

        max_rec = float(rec_raw_all.max()) if rec_raw_all.size > 0 else 0.0
        max_freq = float(freq_all.max()) if freq_all.size > 0 else 0.0

    scored = []
    for num in range(100):
        ml_score = float(ml_probs[num])

        w_num = 0.0
        if prob_mode == "weekly":
            if max_weekly > 0.0:
                w_week = weekly_profile[num, dow_index]
                w_num = w_week / max_weekly
        elif prob_mode == "monthly":
            if max_monthly > 0.0 and 1 <= dd <= 31:
                w_month = monthly_profile[num, dd]
                w_num = w_month / max_monthly
        elif prob_mode == "both":
            vals = []
            if max_weekly > 0.0:
                vals.append(weekly_profile[num, dow_index] / max_weekly)
            if max_monthly > 0.0 and 1 <= dd <= 31:
                vals.append(monthly_profile[num, dd] / max_monthly)
            w_num = sum(vals) / len(vals) if vals else 0.0
        elif prob_mode == "recent":
            if max_rec > 0.0:
                r_norm = rec_raw_all[num] / max_rec
            else:
                r_norm = 0.0
            if max_freq > 0.0:
                f_norm = freq_all[num] / max_freq
            else:
                f_norm = 0.0
            beta = 1.0 - recency_alpha
            w_num = recency_alpha * r_norm + beta * f_norm
        else:
            w_num = 0.0

        if w_num > 0:
            final_score = ml_score * (1.0 - prob_weight_coeff) + w_num * prob_weight_coeff
        else:
            final_score = ml_score

        scored.append(
            {
                "number": num,
                "ml_score": ml_score,
                "prob_score": w_num,
                "final_score": final_score,
            }
        )

    scored_sorted = sorted(scored, key=lambda r: r["final_score"], reverse=True)
    return scored_sorted


# -------------------- Streamlit UI -------------------- #

st.set_page_config(
    page_title="Lottery Tools — Range Wins + ML Ensemble (CatBoost)",
    layout="wide",
)

st.title("Lottery Tools — Range Wins + ML Ensemble (CatBoost)")

st.markdown(
    """
    This Streamlit app mirrors your browser tools:

    1. **Range Wins (per-date)** — classical range/custom analysis + CatBoost model for *Any WIN*.
    2. **ML — Ensemble Number Prediction (0–99)** — CatBoost replacement for your TensorFlow ensemble
       for a *single lottery*.
    3. **ML — Ensemble Number Prediction (0–99, All Lotteries Combined)** — same as (2) but trained on
       the union of **DR, FB, GZ/GB, GL** per date.
    """
)

# ---- Shared CSV upload ---- #
col_up_left, col_up_right = st.columns([2, 1])

with col_up_left:
    uploaded = st.file_uploader("Upload CSV (same format as calcnew.html)", type=["csv", "tsv"])

with col_up_right:
    rmin = st.number_input("Range min (0–99)", min_value=0, max_value=99, value=0)
    rmax = st.number_input("Range max (0–99)", min_value=0, max_value=99, value=9)
    custom_text = st.text_input("Custom numbers (comma/space separated)", placeholder="e.g. 5, 11, 44, 88")
    custom_only = st.checkbox("Custom only (ignore range)")

if rmin > rmax:
    st.warning("Range min > max; swapped internally.")
    rmin, rmax = rmax, rmin

custom_set = parse_custom_nums(custom_text)

# Init session state
if "built_df" not in st.session_state:
    st.session_state["built_df"] = None
    st.session_state["has_gz"] = True
    st.session_state["range"] = (rmin, rmax)
    st.session_state["custom_set"] = custom_set
    st.session_state["custom_only"] = custom_only
    st.session_state["cb_range_model"] = None
    st.session_state["cb_range_metrics"] = None
    st.session_state["ens_model"] = None
    st.session_state["ens_metrics"] = None
    st.session_state["ens_model_config"] = None
    st.session_state["ens_search_history"] = None
    # For ALL lotteries combined
    st.session_state["ens_all_model"] = None
    st.session_state["ens_all_metrics"] = None
    st.session_state["ens_all_model_config"] = None
    st.session_state["ens_all_search_history"] = None

analyze_clicked = st.button("Analyze & Prepare Data")

if analyze_clicked and not uploaded:
    st.error("Please upload a CSV first.")

if uploaded and analyze_clicked:
    try:
        date_map, count_dates, has_gz = load_csv_to_map(uploaded)
        built_df = build_rows_df(date_map, rmin, rmax, custom_set, custom_only)
        st.session_state["built_df"] = built_df
        st.session_state["has_gz"] = has_gz
        st.session_state["range"] = (rmin, rmax)
        st.session_state["custom_set"] = custom_set
        st.session_state["custom_only"] = custom_only
        st.session_state["cb_range_model"] = None
        st.session_state["cb_range_metrics"] = None
        st.session_state["ens_model"] = None
        st.session_state["ens_metrics"] = None
        st.session_state["ens_model_config"] = None
        st.session_state["ens_search_history"] = None
        st.session_state["ens_all_model"] = None
        st.session_state["ens_all_metrics"] = None
        st.session_state["ens_all_model_config"] = None
        st.session_state["ens_all_search_history"] = None
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")

built_df = st.session_state.get("built_df", None)

# Build combined ALL = union of DR, FB, GZ/GB, GL
combined_df = None
if built_df is not None and not built_df.empty:
    def _combine_all(row):
        combined = []
        for k in ["DR", "FB", "GZGB", "GL"]:
            arr = row.get(k, [])
            if isinstance(arr, list):
                combined.extend(arr)
        return combined

    combined_df = built_df.copy()
    combined_df["ALL"] = combined_df.apply(_combine_all, axis=1)

# -------------------- Tabs -------------------- #
tab1, tab2, tab3 = st.tabs(
    [
        "Range Wins + CatBoost (Any WIN)",
        "ML — Ensemble Number Prediction (0–99)",
        "ML — Ensemble (All Lotteries Combined)",
    ]
)

# ==================== TAB 1: Range Wins ==================== #
with tab1:
    if built_df is None or built_df.empty:
        st.info("Upload a CSV and click **Analyze & Prepare Data** to use this section.")
    else:
        rmin, rmax = st.session_state["range"]
        custom_set = st.session_state["custom_set"]
        custom_only = st.session_state["custom_only"]

        st.subheader("Per-day Range Wins")

        summary = summarize_wins(built_df)
        cols = st.columns(5)
        cols[0].metric("Total days", summary["total_days"])
        cols[1].metric("DR WIN days", summary["DR"])
        cols[2].metric("FB WIN days", summary["FB"])
        cols[3].metric("GZ/GB WIN days", summary["GZGB"])
        cols[4].metric("GL WIN days", summary["GL"])
        st.caption(
            f"Any WIN days: **{summary['ANY']}**  • "
            f"Range: `{rmin}-{rmax}`  • Custom: {sorted(list(custom_set)) or 'None'}"
        )

        table_df = pd.DataFrame(
            {
                "#": range(1, len(built_df) + 1),
                "Date": built_df["date_key"],
                "Day": built_df["dow_label"],
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

        dl_df = table_df.copy()
        dl_df["Range"] = f"{rmin}-{rmax}"
        csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download analyzed CSV",
            data=csv_bytes,
            file_name="lottery_range_wins.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Probability Engine (Day-of-Week / Day-of-Month)")

        c1, c2, c3, c4 = st.columns(4)
        mode_label = c1.selectbox("Mode", ["Day-of-Week", "Day-of-Month"])
        alpha = c2.number_input("Alpha (smoothing)", min_value=0.0, value=2.0, step=0.5)
        lam = c3.number_input("Decay λ (recency)", min_value=0.0, value=0.01, step=0.005, format="%0.5f")
        advanced = c4.checkbox("Use advanced blend", value=True)

        if st.button("Run Probability", key="prob_run"):
            mode = "dow" if mode_label == "Day-of-Week" else "dom"
            prob_res = analyze_prob(built_df, mode=mode, alpha=alpha, lam=lam, advanced=advanced)
            if prob_res is None:
                st.warning("No data.")
            else:
                prob_df = pd.DataFrame(prob_res["results"])
                display_df = prob_df.copy()
                for col in ["basic", "smoothed", "weighted", "final"]:
                    display_df[col] = (display_df[col] * 100.0).round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                chart_df = prob_df[["bucket", "final"]].set_index("bucket")
                st.bar_chart(chart_df)

        st.markdown("---")
        st.subheader("CatBoost Model — Any WIN for Range/Custom")

        if not CATBOOST_AVAILABLE:
            st.error("CatBoost is not installed. Install it with `pip install catboost`.")
        else:
            train_ratio = st.slider(
                "Train/Test split (chronological train size)",
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05,
                key="range_train_ratio",
            )

            if st.button("Train CatBoost model (range-level)", key="range_train_btn"):
                with st.spinner("Training CatBoost (range-level)..."):
                    model, metrics = train_catboost_range(
                        built_df, rmin=rmin, rmax=rmax, custom_set=custom_set, train_ratio=train_ratio
                    )
                    st.session_state["cb_range_model"] = model
                    st.session_state["cb_range_metrics"] = metrics

                m = metrics
                mcols = st.columns(5)
                mcols[0].metric("Train samples", m["n_train"])
                mcols[1].metric("Test samples", m["n_test"])
                mcols[2].metric("Accuracy", f"{m['accuracy']*100:.2f}%" if not np.isnan(m["accuracy"]) else "NA")
                mcols[3].metric("Precision", f"{m['precision']*100:.2f}%" if not np.isnan(m["precision"]) else "NA")
                mcols[4].metric("Recall", f"{m['recall']*100:.2f}%" if not np.isnan(m["recall"]) else "NA")
                st.caption(f"AUC: {m['auc']:.4f}" if not np.isnan(m["auc"]) else "AUC: NA")

            cb_model = st.session_state.get("cb_range_model")
            if cb_model is not None:
                st.success("Range-level CatBoost model trained.")
                pred_date = st.date_input(
                    "Pick a date to score Any-WIN probability for this range/custom",
                    value=date.today(),
                    key="range_pred_date",
                )
                if st.button("Predict Any-WIN probability for this date", key="range_pred_btn"):
                    X_new = make_single_features_for_date_range(pred_date, rmin, rmax, custom_set)
                    proba = cb_model.predict_proba(X_new)[:, 1][0]
                    st.metric("Predicted Any-WIN probability", f"{proba*100:.2f}%")
            else:
                st.info("Train the range-level CatBoost model to enable date-wise predictions.")

# ==================== TAB 2: ML Ensemble 0–99 (single lottery) ==================== #
with tab2:
    if built_df is None or built_df.empty:
        st.info("Upload a CSV and click **Analyze & Prepare Data** first. This populates the dataset.")
    elif not CATBOOST_AVAILABLE:
        st.error("CatBoost is not installed. Install it with `pip install catboost` to use the ensemble model.")
    else:
        st.subheader("ML — Ensemble Number Prediction (0–99)")

        col_top_left, col_top_mid, col_top_right = st.columns(3)
        lottery_choice = col_top_left.selectbox(
            "Lottery",
            ["DR", "FB", "GZ/GB", "GL"],
            index=0,
        )

        prob_mode_label = col_top_mid.radio(
            "Probability Weight Strategy",
            ["Weekly only", "Monthly only", "Weekly + Monthly (avg)", "Recency-weighted (3-month focus)"],
        )

        prob_weight = col_top_right.number_input(
            "ML / Probability blending (0 = pure ML, 1 = pure probability)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
        )

        lot_key_map = {
            "DR": "DR",
            "FB": "FB",
            "GZ/GB": "GZGB",
            "GL": "GL",
        }
        lottery_key = lot_key_map[lottery_choice]

        st.markdown(
            "*Default 0.4 = 60% ML + 40% probability, exactly matching your browser slider semantics.*"
        )

        recency_half_life = 30.0
        recency_window_days = 90.0
        recency_alpha = 0.7
        if prob_mode_label == "Recency-weighted (3-month focus)":
            with st.expander("Recency settings (advanced)", expanded=False):
                recency_half_life = st.number_input(
                    "Recency half-life (days)",
                    min_value=1.0,
                    max_value=365.0,
                    value=30.0,
                    step=1.0,
                    key="rec_half_life",
                )
                recency_window_days = st.number_input(
                    "Recency window (days)",
                    min_value=7.0,
                    max_value=365.0,
                    value=90.0,
                    step=1.0,
                    key="rec_window_days",
                )
                recency_alpha = st.slider(
                    "Recency weight α (recency vs frequency)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="rec_alpha",
                )

        prob_mode_map = {
            "Weekly only": "weekly",
            "Monthly only": "monthly",
            "Weekly + Monthly (avg)": "both",
            "Recency-weighted (3-month focus)": "recent",
        }
        prob_mode = prob_mode_map[prob_mode_label]

        st.markdown("---")
        st.markdown("**Core / Mid / Edge sizing** — `Core + Mid ≤ Top N` (Edge = remaining).")
        c1, c2, c3 = st.columns(3)
        top_n = c1.number_input("Top N numbers", min_value=1, max_value=100, value=20, step=1)
        core_count = c2.number_input("Core size", min_value=0, max_value=100, value=8, step=1)
        mid_count = c3.number_input("Mid size", min_value=0, max_value=100, value=6, step=1)

        if core_count + mid_count > top_n:
            mid_count = max(0, top_n - core_count)
            st.warning(f"Adjusted Mid to {mid_count} so Core + Mid ≤ Top N.")

        edge_count = max(0, top_n - core_count - mid_count)
        st.caption(
            f"Core: **{core_count}**, Mid: **{mid_count}**, Edge (within Top N): **{edge_count}**."
        )

        st.markdown("---")
        st.subheader("1) Train Ensemble Model (CatBoost replacement)")

        ens_train_ratio = st.slider(
            "Train/Validation split (chronological)",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            key="ens_train_ratio",
        )

        if st.button("Train Ensemble Model (CatBoost)", key="ens_train_btn"):
            with st.spinner("Training CatBoost ensemble model (0–99)..."):
                model, metrics = ens_train_catboost(
                    built_df,
                    lottery_key,
                    train_ratio=ens_train_ratio,
                    recency_half_life=recency_half_life,
                    recency_window_days=recency_window_days,
                )
                st.session_state["ens_model"] = model
                st.session_state["ens_metrics"] = metrics
                st.session_state["ens_model_config"] = {
                    "train_ratio": ens_train_ratio,
                    "recency_half_life": recency_half_life,
                    "recency_window_days": recency_window_days,
                    "val_accuracy": metrics.get("val_accuracy", float("nan")),
                    "val_auc": metrics.get("val_auc", float("nan")),
                    "samples": metrics.get("samples", np.nan),
                    "n_train": metrics.get("n_train", np.nan),
                    "n_val": metrics.get("n_val", np.nan),
                    "pos_frac_train": metrics.get("pos_frac_train", np.nan),
                    "scale_pos_weight": metrics.get("scale_pos_weight", np.nan),
                    "source": "manual",
                }
                st.session_state["ens_search_history"] = None

            m = metrics
            mcols = st.columns(4)
            mcols[0].metric("Samples", m["samples"])
            mcols[1].metric("Train samples", m["n_train"])
            mcols[2].metric("Val samples", m["n_val"])
            mcols[3].metric(
                "Val accuracy",
                f"{m['val_accuracy']*100:.2f}%" if not np.isnan(m["val_accuracy"]) else "NA",
            )
            st.caption(
                f"AUC: {m['val_auc']:.4f}  •  "
                f"Pos frac (train): {m['pos_frac_train']:.4f}  •  "
                f"scale_pos_weight: {m['scale_pos_weight']:.2f}"
            )

        if st.button("Auto-search best Ensemble model (max val accuracy)", key="ens_auto_btn"):
            with st.spinner("Searching over train split + recency hyperparameters..."):
                best_model, best_config, hist_df = ens_auto_search_best_model(
                    built_df,
                    lottery_key=lottery_key,
                )
                if best_model is not None and best_config is not None:
                    st.session_state["ens_model"] = best_model
                    st.session_state["ens_model_config"] = best_config
                    st.session_state["ens_search_history"] = hist_df
                    st.session_state["ens_metrics"] = {
                        "samples": best_config.get("samples", np.nan),
                        "n_train": best_config.get("n_train", np.nan),
                        "n_val": best_config.get("n_val", np.nan),
                        "val_accuracy": best_config.get("val_accuracy", float("nan")),
                        "val_auc": best_config.get("val_auc", float("nan")),
                        "pos_frac_train": best_config.get("pos_frac_train", np.nan),
                        "scale_pos_weight": best_config.get("scale_pos_weight", np.nan),
                    }

            best_cfg = st.session_state.get("ens_model_config")
            if best_cfg is None:
                st.warning("Auto-search did not produce a valid model.")
            else:
                st.success(
                    f"Best val accuracy: {best_cfg['val_accuracy']*100:.2f}%  •  "
                    f"train_ratio={best_cfg['train_ratio']}, "
                    f"half_life={best_cfg['recency_half_life']}, "
                    f"window={best_cfg['recency_window_days']}"
                )
                hist_df = st.session_state.get("ens_search_history")
                if hist_df is not None and not hist_df.empty:
                    st.markdown("**Hyperparameter search history (sorted by val_accuracy):**")
                    st.dataframe(
                        hist_df.sort_values("val_accuracy", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

        ens_model = st.session_state.get("ens_model")
        ens_cfg = st.session_state.get("ens_model_config")

        if ens_cfg is not None:
            st.caption(
                f"Current ensemble model source: {ens_cfg.get('source', 'unknown')}  •  "
                f"train_ratio={ens_cfg.get('train_ratio')}  •  "
                f"half_life={ens_cfg.get('recency_half_life')}  •  "
                f"window_days={ens_cfg.get('recency_window_days')}  •  "
                f"val_acc={ens_cfg.get('val_accuracy', float('nan')):.3f}"
            )

        st.markdown("---")
        st.subheader("2) Score numbers 0–99 for a prediction date")

        if ens_model is None:
            st.info("Train or auto-search the ensemble model first.")
        else:
            eff_rec_half = recency_half_life
            eff_rec_window = recency_window_days
            if ens_cfg is not None:
                eff_rec_half = float(ens_cfg.get("recency_half_life", eff_rec_half))
                eff_rec_window = float(ens_cfg.get("recency_window_days", eff_rec_window))

            last_date = built_df["date"].max().date()
            default_pred_date = last_date + timedelta(days=1)
            pred_date = st.date_input(
                "Prediction date (for ranking 0–99)",
                value=default_pred_date,
                key="ens_pred_date",
            )

            if st.button("Score numbers (Core / Mid / Edge)", key="ens_score_btn"):
                history_mask = built_df["date"].dt.date < pred_date
                history_df = built_df.loc[history_mask].copy()

                if history_df.empty:
                    st.warning("No historical data before this prediction date.")
                else:
                    with st.spinner("Scoring numbers 0–99 with ML + probability fusion..."):
                        scored_sorted = ens_score_numbers_for_date(
                            ens_model,
                            history_df,
                            lottery_key=lottery_key,
                            prob_mode=prob_mode,
                            prob_weight_coeff=float(prob_weight),
                            prediction_date=pred_date,
                            recency_half_life=eff_rec_half,
                            recency_window_days=eff_rec_window,
                            recency_alpha=recency_alpha,
                        )

                    top_list = scored_sorted[: int(top_n)]
                    rows = []
                    for idx, item in enumerate(top_list):
                        num = item["number"]
                        ml_score = item["ml_score"]
                        prob_score = item["prob_score"]
                        final_score = item["final_score"]

                        if idx < core_count:
                            tier = "Core"
                        elif idx < core_count + mid_count:
                            tier = "Mid"
                        else:
                            tier = "Edge"

                        rows.append(
                            {
                                "Rank": idx + 1,
                                "Number": num,
                                "Tier": tier,
                                "Final Score": round(final_score, 6),
                                "ML Score": round(ml_score, 6),
                                "Prob Score": round(prob_score, 6),
                            }
                        )

                    result_df = pd.DataFrame(rows)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Top-N (Core/Mid/Edge) as CSV",
                        data=csv_bytes,
                        file_name="ml_ensemble_top_numbers.csv",
                        mime="text/csv",
                    )

                    st.success(
                        f"Scored numbers 0–99 for {pred_date.isoformat()} "
                        f"using **{lottery_choice}**, prob mode **{prob_mode_label}**, "
                        f"blend={prob_weight:.2f}."
                    )

        if ens_model is not None:
            st.markdown("---")
            st.subheader("3) Backtest ensemble vs actual results (date range)")

            eval_min_date = built_df["date"].min().date()
            eval_max_date = built_df["date"].max().date()

            eval_date_sel = st.date_input(
                "Evaluation date range (for backtesting)",
                (eval_min_date, eval_max_date),
                min_value=eval_min_date,
                max_value=eval_max_date,
                key="ens_eval_range",
            )

            if isinstance(eval_date_sel, tuple) and len(eval_date_sel) == 2:
                eval_start, eval_end = eval_date_sel
            else:
                eval_start = eval_end = eval_date_sel

            if eval_start > eval_end:
                st.error("Evaluation start date is after end date.")
            else:
                eff_rec_half_bt = recency_half_life
                eff_rec_window_bt = recency_window_days
                if ens_cfg is not None:
                    eff_rec_half_bt = float(ens_cfg.get("recency_half_life", eff_rec_half_bt))
                    eff_rec_window_bt = float(ens_cfg.get("recency_window_days", eff_rec_window_bt))

                if st.button("Run ensemble backtest", key="ens_backtest_btn"):
                    total_calendar_days = (eval_end - eval_start).days + 1
                    dates_with_data = 0
                    total_hits_all = 0
                    core_hits_total = 0
                    core_mid_hits_total = 0

                    detail_rows = []

                    for offset in range(total_calendar_days):
                        d = eval_start + timedelta(days=offset)

                        mask_day = built_df["date"].dt.date == d
                        if not mask_day.any():
                            continue

                        row = built_df.loc[mask_day].iloc[0]
                        actual_arr = row[lottery_key] if isinstance(row[lottery_key], list) else []
                        actual_set = set()
                        for x in actual_arr:
                            try:
                                v = int(x)
                            except Exception:
                                continue
                            if 0 <= v <= 99:
                                actual_set.add(v)

                        if not actual_set:
                            continue

                        history_mask = built_df["date"].dt.date < d
                        history_df = built_df.loc[history_mask].copy()
                        if history_df.empty:
                            continue

                        dates_with_data += 1

                        scored_sorted = ens_score_numbers_for_date(
                            ens_model,
                            history_df,
                            lottery_key=lottery_key,
                            prob_mode=prob_mode,
                            prob_weight_coeff=float(prob_weight),
                            prediction_date=d,
                            recency_half_life=eff_rec_half_bt,
                            recency_window_days=eff_rec_window_bt,
                            recency_alpha=recency_alpha,
                        )

                        top_list = scored_sorted[: int(top_n)]

                        core_hits = 0
                        mid_hits = 0
                        edge_hits = 0

                        for idx, item in enumerate(top_list):
                            num = item["number"]
                            if num in actual_set:
                                if idx < core_count:
                                    core_hits += 1
                                elif idx < core_count + mid_count:
                                    mid_hits += 1
                                else:
                                    edge_hits += 1

                        day_total_hits = core_hits + mid_hits + edge_hits
                        total_hits_all += day_total_hits
                        core_hits_total += core_hits
                        core_mid_hits_total += core_hits + mid_hits

                        detail_rows.append(
                            {
                                "Date": d.isoformat(),
                                "Day": row["dow_label"],
                                "Actual": " ".join(str(n) for n in sorted(actual_set)),
                                "Core Hits": core_hits,
                                "Mid Hits": mid_hits,
                                "Edge Hits": edge_hits,
                                "Total Hits": day_total_hits,
                            }
                        )

                    if dates_with_data == 0:
                        st.warning("No dates with actual data found in this range for backtesting.")
                    else:
                        avg_hits_per_date = total_hits_all / dates_with_data
                        core_hit_rate = (core_hits_total / dates_with_data) * 100.0
                        core_mid_hit_rate = (core_mid_hits_total / dates_with_data) * 100.0
                        any_tier_hit_rate = (total_hits_all / dates_with_data) * 100.0

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Dates in range", total_calendar_days)
                        m2.metric("Dates with actual data", dates_with_data)
                        m3.metric("Total hits (all tiers)", total_hits_all)
                        m4.metric("Avg hits / date (past only)", f"{avg_hits_per_date:.2f}")

                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric("Core hit rate", f"{core_hit_rate:.1f}%")
                        n2.metric("Core+Mid hit rate", f"{core_mid_hit_rate:.1f}%")
                        n3.metric("Any tier hit rate", f"{any_tier_hit_rate:.1f}%")
                        n4.metric("Lottery", lottery_choice)

                        if detail_rows:
                            detail_df = pd.DataFrame(detail_rows)
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)

                            csv_bytes = detail_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download backtest details as CSV",
                                data=csv_bytes,
                                file_name="ensemble_backtest_details.csv",
                                mime="text/csv",
                            )

# ==================== TAB 3: ML Ensemble 0–99 (ALL lotteries combined) ==================== #
with tab3:
    if combined_df is None or combined_df.empty:
        st.info("Upload a CSV and click **Analyze & Prepare Data** first. This populates the dataset.")
    elif not CATBOOST_AVAILABLE:
        st.error("CatBoost is not installed. Install it with `pip install catboost` to use the ensemble model.")
    else:
        st.subheader("ML — Ensemble Number Prediction (0–99, All Lotteries Combined)")

        col_top_mid, col_top_right = st.columns(2)

        prob_mode_label_all = col_top_mid.radio(
            "Probability Weight Strategy",
            ["Weekly only", "Monthly only", "Weekly + Monthly (avg)", "Recency-weighted (3-month focus)"],
            key="prob_mode_all",
        )

        prob_weight_all = col_top_right.number_input(
            "ML / Probability blending (0 = pure ML, 1 = pure probability)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key="prob_weight_all",
        )

        lottery_key_all = "ALL"
        st.markdown(
            "*Trained on the union of **DR, FB, GZ/GB, GL** for each date (Any lottery hit counts as a hit).*"
        )

        # --- Recency options (same semantics as single-lottery tab) ---
        recency_half_life_all = 30.0
        recency_window_days_all = 90.0
        recency_alpha_all = 0.7
        if prob_mode_label_all == "Recency-weighted (3-month focus)":
            with st.expander("Recency settings (advanced)", expanded=False):
                recency_half_life_all = st.number_input(
                    "Recency half-life (days)",
                    min_value=1.0,
                    max_value=365.0,
                    value=30.0,
                    step=1.0,
                    key="rec_half_life_all",
                )
                recency_window_days_all = st.number_input(
                    "Recency window (days)",
                    min_value=7.0,
                    max_value=365.0,
                    value=90.0,
                    step=1.0,
                    key="rec_window_days_all",
                )
                recency_alpha_all = st.slider(
                    "Recency weight α (recency vs frequency)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="rec_alpha_all",
                )

        prob_mode_map_all = {
            "Weekly only": "weekly",
            "Monthly only": "monthly",
            "Weekly + Monthly (avg)": "both",
            "Recency-weighted (3-month focus)": "recent",
        }
        prob_mode_all = prob_mode_map_all[prob_mode_label_all]

        st.markdown("---")
        st.markdown("**Core / Mid / Edge sizing** — `Core + Mid ≤ Top N` (Edge = remaining).")
        c1_all, c2_all, c3_all = st.columns(3)
        top_n_all = c1_all.number_input("Top N numbers (combined)", min_value=1, max_value=100, value=20, step=1)
        core_count_all = c2_all.number_input("Core size (combined)", min_value=0, max_value=100, value=8, step=1)
        mid_count_all = c3_all.number_input("Mid size (combined)", min_value=0, max_value=100, value=6, step=1)

        if core_count_all + mid_count_all > top_n_all:
            mid_count_all = max(0, top_n_all - core_count_all)
            st.warning(f"Adjusted Mid (combined) to {mid_count_all} so Core + Mid ≤ Top N.")

        edge_count_all = max(0, top_n_all - core_count_all - mid_count_all)
        st.caption(
            f"(Combined) Core: **{core_count_all}**, Mid: **{mid_count_all}**, Edge (within Top N): **{edge_count_all}**."
        )

        st.markdown("---")
        st.subheader("1) Train Combined Ensemble Model (CatBoost replacement)")

        ens_all_train_ratio = st.slider(
            "Train/Validation split (chronological, combined)",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            key="ens_all_train_ratio",
        )

        # Manual combined training (for forward predictions / inspection)
        if st.button("Train Combined Ensemble Model (CatBoost)", key="ens_all_train_btn"):
            with st.spinner("Training combined CatBoost ensemble model (0–99, all lotteries)..."):
                model_all, metrics_all = ens_train_catboost(
                    combined_df,
                    lottery_key_all,
                    train_ratio=ens_all_train_ratio,
                    recency_half_life=recency_half_life_all,
                    recency_window_days=recency_window_days_all,
                )
                st.session_state["ens_all_model"] = model_all
                st.session_state["ens_all_metrics"] = metrics_all
                st.session_state["ens_all_model_config"] = {
                    "train_ratio": ens_all_train_ratio,
                    "recency_half_life": recency_half_life_all,
                    "recency_window_days": recency_window_days_all,
                    "val_accuracy": metrics_all.get("val_accuracy", float("nan")),
                    "val_auc": metrics_all.get("val_auc", float("nan")),
                    "samples": metrics_all.get("samples", np.nan),
                    "n_train": metrics_all.get("n_train", np.nan),
                    "n_val": metrics_all.get("n_val", np.nan),
                    "pos_frac_train": metrics_all.get("pos_frac_train", np.nan),
                    "scale_pos_weight": metrics_all.get("scale_pos_weight", np.nan),
                    "source": "manual_all",
                }
                st.session_state["ens_all_search_history"] = None

            m_all = metrics_all
            mcols_all = st.columns(4)
            mcols_all[0].metric("Samples", m_all["samples"])
            mcols_all[1].metric("Train samples", m_all["n_train"])
            mcols_all[2].metric("Val samples", m_all["n_val"])
            mcols_all[3].metric(
                "Val accuracy",
                f"{m_all['val_accuracy']*100:.2f}%" if not np.isnan(m_all["val_accuracy"]) else "NA",
            )
            st.caption(
                f"AUC: {m_all['val_auc']:.4f}  •  "
                f"Pos frac (train): {m_all['pos_frac_train']:.4f}  •  "
                f"scale_pos_weight: {m_all['scale_pos_weight']:.2f}"
            )

        # Auto-search best combined model config (global)
        if st.button("Auto-search best Combined Ensemble model (max val accuracy)", key="ens_all_auto_btn"):
            with st.spinner("Searching over train split + recency hyperparameters (combined)..."):
                best_model_all, best_config_all, hist_df_all = ens_auto_search_best_model(
                    combined_df,
                    lottery_key=lottery_key_all,
                )
                if best_model_all is not None and best_config_all is not None:
                    st.session_state["ens_all_model"] = best_model_all
                    st.session_state["ens_all_model_config"] = best_config_all
                    st.session_state["ens_all_search_history"] = hist_df_all
                    st.session_state["ens_all_metrics"] = {
                        "samples": best_config_all.get("samples", np.nan),
                        "n_train": best_config_all.get("n_train", np.nan),
                        "n_val": best_config_all.get("n_val", np.nan),
                        "val_accuracy": best_config_all.get("val_accuracy", float("nan")),
                        "val_auc": best_config_all.get("val_auc", float("nan")),
                        "pos_frac_train": best_config_all.get("pos_frac_train", np.nan),
                        "scale_pos_weight": best_config_all.get("scale_pos_weight", np.nan),
                    }

            best_cfg_all = st.session_state.get("ens_all_model_config")
            if best_cfg_all is None:
                st.warning("Auto-search did not produce a valid combined model.")
            else:
                st.success(
                    f"[Combined] Best val accuracy: {best_cfg_all['val_accuracy']*100:.2f}%  •  "
                    f"train_ratio={best_cfg_all['train_ratio']}, "
                    f"half_life={best_cfg_all['recency_half_life']}, "
                    f"window={best_cfg_all['recency_window_days']}"
                )
                hist_df_all = st.session_state.get("ens_all_search_history")
                if hist_df_all is not None and not hist_df_all.empty:
                    st.markdown("**[Combined] Hyperparameter search history (sorted by val_accuracy):**")
                    st.dataframe(
                        hist_df_all.sort_values("val_accuracy", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

        ens_all_model = st.session_state.get("ens_all_model")
        ens_all_cfg = st.session_state.get("ens_all_model_config")

        if ens_all_cfg is not None:
            st.caption(
                f"[Combined] Current ensemble model source: {ens_all_cfg.get('source', 'unknown')}  •  "
                f"train_ratio={ens_all_cfg.get('train_ratio')}  •  "
                f"half_life={ens_all_cfg.get('recency_half_life')}  •  "
                f"window_days={ens_all_cfg.get('recency_window_days')}  •  "
                f"val_acc={ens_all_cfg.get('val_accuracy', float('nan')):.3f}"
            )

        st.markdown("---")
        st.subheader("2) Score numbers 0–99 for a prediction date (combined)")

        if ens_all_model is None:
            st.info("Train or auto-search the combined ensemble model first.")
        else:
            eff_rec_half_all = recency_half_life_all
            eff_rec_window_all = recency_window_days_all
            if ens_all_cfg is not None:
                eff_rec_half_all = float(ens_all_cfg.get("recency_half_life", eff_rec_half_all))
                eff_rec_window_all = float(ens_all_cfg.get("recency_window_days", eff_rec_window_all))

            last_date_all = combined_df["date"].max().date()
            default_pred_date_all = last_date_all + timedelta(days=1)
            pred_date_all = st.date_input(
                "Prediction date (for ranking 0–99, combined)",
                value=default_pred_date_all,
                key="ens_all_pred_date",
            )

            if st.button("Score numbers (Core / Mid / Edge, combined)", key="ens_all_score_btn"):
                history_mask_all = combined_df["date"].dt.date < pred_date_all
                history_df_all = combined_df.loc[history_mask_all].copy()

                if history_df_all.empty:
                    st.warning("No historical data before this prediction date (combined).")
                else:
                    with st.spinner("Scoring numbers 0–99 with ML + probability fusion (combined)..."):
                        scored_sorted_all = ens_score_numbers_for_date(
                            ens_all_model,
                            history_df_all,
                            lottery_key=lottery_key_all,
                            prob_mode=prob_mode_all,
                            prob_weight_coeff=float(prob_weight_all),
                            prediction_date=pred_date_all,
                            recency_half_life=eff_rec_half_all,
                            recency_window_days=eff_rec_window_all,
                            recency_alpha=recency_alpha_all,
                        )

                    top_list_all = scored_sorted_all[: int(top_n_all)]
                    rows_all = []
                    for idx, item in enumerate(top_list_all):
                        num = item["number"]
                        ml_score = item["ml_score"]
                        prob_score = item["prob_score"]
                        final_score = item["final_score"]

                        if idx < core_count_all:
                            tier = "Core"
                        elif idx < core_count_all + mid_count_all:
                            tier = "Mid"
                        else:
                            tier = "Edge"

                        rows_all.append(
                            {
                                "Rank": idx + 1,
                                "Number": num,
                                "Tier": tier,
                                "Final Score": round(final_score, 6),
                                "ML Score": round(ml_score, 6),
                                "Prob Score": round(prob_score, 6),
                            }
                        )

                    result_df_all = pd.DataFrame(rows_all)
                    st.dataframe(result_df_all, use_container_width=True, hide_index=True)

                    csv_bytes_all = result_df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Top-N (Core/Mid/Edge, combined) as CSV",
                        data=csv_bytes_all,
                        file_name="ml_ensemble_top_numbers_combined.csv",
                        mime="text/csv",
                    )

                    st.success(
                        f"[Combined] Scored numbers 0–99 for {pred_date_all.isoformat()} "
                        f"using **ALL lotteries**, prob mode **{prob_mode_label_all}**, "
                        f"blend={prob_weight_all:.2f}."
                    )

        # --- NEW: Strict walk-forward backtest (no future leakage) ---
        st.markdown("---")
        st.subheader("3) Backtest combined ensemble vs actual results (walk-forward, no future leakage)")

        eval_min_date_all = combined_df["date"].min().date()
        eval_max_date_all = combined_df["date"].max().date()

        eval_date_sel_all = st.date_input(
            "Evaluation date range (for backtesting, combined)",
            (eval_min_date_all, eval_max_date_all),
            min_value=eval_min_date_all,
            max_value=eval_max_date_all,
            key="ens_all_eval_range",
        )

        if isinstance(eval_date_sel_all, tuple) and len(eval_date_sel_all) == 2:
            eval_start_all, eval_end_all = eval_date_sel_all
        else:
            eval_start_all = eval_end_all = eval_date_sel_all

        if eval_start_all > eval_end_all:
            st.error("Evaluation start date is after end date (combined).")
        else:
            # Use trained config if available; otherwise fall back to current UI settings
            eff_rec_half_bt_all = recency_half_life_all
            eff_rec_window_bt_all = recency_window_days_all
            train_ratio_bt_all = ens_all_train_ratio
            if ens_all_cfg is not None:
                eff_rec_half_bt_all = float(ens_all_cfg.get("recency_half_life", eff_rec_half_bt_all))
                eff_rec_window_bt_all = float(ens_all_cfg.get("recency_window_days", eff_rec_window_bt_all))
                train_ratio_bt_all = float(ens_all_cfg.get("train_ratio", train_ratio_bt_all))

            if st.button("Run combined ensemble backtest (walk-forward)", key="ens_all_backtest_btn"):
                total_calendar_days_all = (eval_end_all - eval_start_all).days + 1
                dates_with_data_all = 0
                total_hits_all_all = 0
                core_hits_total_all = 0
                core_mid_hits_total_all = 0

                detail_rows_all = []

                for offset in range(total_calendar_days_all):
                    d_all = eval_start_all + timedelta(days=offset)

                    # Actual combined results for day d_all
                    mask_day_all = combined_df["date"].dt.date == d_all
                    if not mask_day_all.any():
                        continue

                    row_all = combined_df.loc[mask_day_all].iloc[0]
                    actual_arr_all = row_all[lottery_key_all] if isinstance(row_all[lottery_key_all], list) else []
                    actual_set_all = set()
                    for x in actual_arr_all:
                        try:
                            v = int(x)
                        except Exception:
                            continue
                        if 0 <= v <= 99:
                            actual_set_all.add(v)

                    if not actual_set_all:
                        continue

                    # History strictly before d_all
                    history_mask_all_bt = combined_df["date"].dt.date < d_all
                    history_df_all_bt = combined_df.loc[history_mask_all_bt].copy()
                    if history_df_all_bt.empty:
                        continue

                    # Train a fresh model on history only (no future leakage)
                    try:
                        model_d, metrics_d = ens_train_catboost(
                            history_df_all_bt,
                            lottery_key=lottery_key_all,
                            train_ratio=train_ratio_bt_all,
                            recency_half_life=eff_rec_half_bt_all,
                            recency_window_days=eff_rec_window_bt_all,
                        )
                    except Exception:
                        # If model training fails (edge cases), skip this date
                        continue

                    dates_with_data_all += 1

                    # Score this date using only that history + newly trained model
                    scored_sorted_all_bt = ens_score_numbers_for_date(
                        model_d,
                        history_df_all_bt,
                        lottery_key=lottery_key_all,
                        prob_mode=prob_mode_all,
                        prob_weight_coeff=float(prob_weight_all),
                        prediction_date=d_all,
                        recency_half_life=eff_rec_half_bt_all,
                        recency_window_days=eff_rec_window_bt_all,
                        recency_alpha=recency_alpha_all,
                    )

                    top_list_all_bt = scored_sorted_all_bt[: int(top_n_all)]

                    core_hits_all = 0
                    mid_hits_all = 0
                    edge_hits_all = 0

                    for idx, item in enumerate(top_list_all_bt):
                        num = item["number"]
                        if num in actual_set_all:
                            if idx < core_count_all:
                                core_hits_all += 1
                            elif idx < core_count_all + mid_count_all:
                                mid_hits_all += 1
                            else:
                                edge_hits_all += 1

                    day_total_hits_all = core_hits_all + mid_hits_all + edge_hits_all
                    total_hits_all_all += day_total_hits_all
                    core_hits_total_all += core_hits_all
                    core_mid_hits_total_all += core_hits_all + mid_hits_all

                    detail_rows_all.append(
                        {
                            "Date": d_all.isoformat(),
                            "Day": row_all["dow_label"],
                            "Actual (ALL lotteries)": " ".join(str(n) for n in sorted(actual_set_all)),
                            "Core Hits": core_hits_all,
                            "Mid Hits": mid_hits_all,
                            "Edge Hits": edge_hits_all,
                            "Total Hits": day_total_hits_all,
                        }
                    )

                if dates_with_data_all == 0:
                    st.warning("No dates with actual data / trainable history for combined backtesting.")
                else:
                    avg_hits_per_date_all = total_hits_all_all / dates_with_data_all
                    core_hit_rate_all = (core_hits_total_all / dates_with_data_all) * 100.0
                    core_mid_hit_rate_all = (core_mid_hits_total_all / dates_with_data_all) * 100.0
                    any_tier_hit_rate_all = (total_hits_all_all / dates_with_data_all) * 100.0

                    m1_all, m2_all, m3_all, m4_all = st.columns(4)
                    m1_all.metric("Dates in range", total_calendar_days_all)
                    m2_all.metric("Dates with data (trainable)", dates_with_data_all)
                    m3_all.metric("Total hits (all tiers, combined)", total_hits_all_all)
                    m4_all.metric("Avg hits / date (combined)", f"{avg_hits_per_date_all:.2f}")

                    n1_all, n2_all, n3_all, n4_all = st.columns(4)
                    n1_all.metric("Core hit rate (combined)", f"{core_hit_rate_all:.1f}%")
                    n2_all.metric("Core+Mid hit rate (combined)", f"{core_mid_hit_rate_all:.1f}%")
                    n3_all.metric("Any tier hit rate (combined)", f"{any_tier_hit_rate_all:.1f}%")
                    n4_all.metric("Lottery", "ALL (combined)")

                    if detail_rows_all:
                        detail_df_all = pd.DataFrame(detail_rows_all)
                        st.dataframe(detail_df_all, use_container_width=True, hide_index=True)

                        csv_bytes_all_bt = detail_df_all.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download combined backtest details as CSV",
                            data=csv_bytes_all_bt,
                            file_name="ensemble_backtest_details_combined_walkforward.csv",
                            mime="text/csv",
                        )

# ============================================================
# Parity / Range Models (Strategy B & Strategy C, per lottery)
# ============================================================

import io
from datetime import timedelta  # used for "next draw" default date

# ---------- Small helpers (local to this section) ---------- #

WINDOWS_PARITY_RANGE = [5, 10, 20, 50]


def detect_delimiter_simple(first_line: str) -> str:
    """
    Simple delimiter detection for the classifier CSV uploader.
    Doesn't touch your existing detect_delimiter().
    """
    if "\t" in first_line:
        return "\t"
    if ";" in first_line:
        return ";"
    if "|" in first_line:
        return "|"
    return ","  # default


def load_csv_robust_for_classifiers(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader:
    - decodes as UTF-8 (ignoring bad chars)
    - auto-detects delimiter
    - returns a pandas DataFrame
    """
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("Uploaded CSV is empty.")

    lines = text.splitlines()
    first_line = lines[0]
    delimiter = detect_delimiter_simple(first_line)
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    return df


def ensure_draw_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to create/standardize a 'draw_date' column.
    Handles:
    - a single 'date'/'drawdate'/'draw_date'/'draw date' column, or
    - year/month/day style columns, or
    - falls back to synthetic chronological index if needed.
    Returns a *new* DataFrame with 'draw_date' and sorted by it.
    """
    df = df.copy()

    # Already present?
    if "draw_date" in df.columns:
        df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce", dayfirst=True)
    else:
        # Try a single date-like column
        date_col = None
        for col in df.columns:
            c = col.strip().lower()
            if c in ("date", "drawdate", "draw_date", "draw date"):
                date_col = col
                break

        if date_col is not None:
            df["draw_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        else:
            # Try year / month / day triplet
            year_col = None
            month_col = None
            day_col = None
            for col in df.columns:
                c = col.strip().lower()
                if c in ("year", "yyyy"):
                    year_col = col
                elif c in ("month", "mm"):
                    month_col = col
                elif c in ("day", "dd", "dom"):
                    day_col = col

            if year_col and month_col and day_col:
                temp = df[[year_col, month_col, day_col]].rename(
                    columns={year_col: "year", month_col: "month", day_col: "day"}
                )
                df["draw_date"] = pd.to_datetime(temp, errors="coerce")
            else:
                # Absolute fallback: synthetic date based on row index
                base = pd.Timestamp("2000-01-01")
                df["draw_date"] = base + pd.to_timedelta(range(len(df)), unit="D")

    # Drop rows where date couldn't be parsed at all
    df = df[df["draw_date"].notna()].copy()
    df = df.sort_values("draw_date").reset_index(drop=True)
    return df


def get_lottery_columns(df: pd.DataFrame) -> list:
    """
    Detect known lottery columns (DR, FB, SG, GZ, GL, GB).
    Returns the *actual* column names as they appear in df.
    """
    KNOWN = {"DR", "FB", "SG", "GZ", "GL", "GB"}
    found = []
    for col in df.columns:
        if col.strip().upper() in KNOWN:
            found.append(col)
    return found


def compute_streak(values: pd.Series) -> pd.Series:
    """
    Compute consecutive-ones streak for a 0/1 Series.
    Example: [0,1,1,0,1] -> [0,1,2,0,1]
    """
    streak = []
    current = 0
    for v in values.fillna(0).astype(int):
        if v == 1:
            current += 1
        else:
            current = 0
        streak.append(current)
    return pd.Series(streak, index=values.index)


def prepare_lottery_dataset_for_parity_range(
    df_raw: pd.DataFrame,
    lottery_col: str,
) -> tuple[pd.DataFrame, list]:
    """
    From the full raw df + chosen lottery column, build:
    - clean 'draw_date'
    - numeric lottery column ('lottery_value') 0-99
    - targets:
        * target_parity: 0 = even, 1 = odd
        * target_range:  0 = low (0-49), 1 = high (50-99)
        * target_4class: 0..3 as:
            0: Even & Low  (0-49 even)
            1: Odd  & Low  (0-49 odd)
            2: Even & High (50-99 even)
            3: Odd  & High (50-99 odd)
    - leak-free features based ONLY on PAST draws:
        * rolling odd/high ratios
        * previous odd/high flag
        * previous odd/high streaks
        * calendar features
    Returns:
        df_feat, feature_cols
    """
    df = ensure_draw_date_column(df_raw)

    if lottery_col not in df.columns:
        raise KeyError(f"Lottery column '{lottery_col}' not found in CSV.")

    # Clean numeric lottery values (0-99)
    lot = pd.to_numeric(df[lottery_col], errors="coerce")
    df = df.loc[lot.notna()].copy()
    df["lottery_value"] = lot.loc[df.index].astype(int)

    # Sort strictly by date again (after filtering)
    df = df.sort_values("draw_date").reset_index(drop=True)

    # ---------- Targets ---------- #
    parity = (df["lottery_value"] % 2).astype(int)       # 0 even, 1 odd
    high   = (df["lottery_value"] >= 50).astype(int)     # 0 low, 1 high

    df["target_parity"] = parity
    df["target_range"]  = high
    # Combine into 4 classes:
    # 0: even+low, 1: odd+low, 2: even+high, 3: odd+high
    df["target_4class"] = parity + 2 * high

    # ---------- Leak-free historical features ---------- #
    # Use only PAST info via shift(1).

    odd_prev  = parity.shift(1).fillna(0).astype(int)
    high_prev = high.shift(1).fillna(0).astype(int)

    df["feat_odd_prev"]  = odd_prev
    df["feat_high_prev"] = high_prev

    # Rolling ratios of odd/high over windows of 5, 10, 20, 50 past draws
    for w in WINDOWS_PARITY_RANGE:
        df[f"feat_odd_ratio_{w}"] = (
            parity.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"feat_high_ratio_{w}"] = (
            high.shift(1).rolling(window=w, min_periods=1).mean()
        )

    # Streaks of past odds/high numbers (in draws, not days)
    df["feat_odd_streak_prev"]  = compute_streak(odd_prev)
    df["feat_high_streak_prev"] = compute_streak(high_prev)

    # ---------- Calendar features ---------- #
    dt = df["draw_date"]
    df["feat_year"]           = dt.dt.year
    df["feat_month"]          = dt.dt.month
    df["feat_day"]            = dt.dt.day
    df["feat_dow"]            = dt.dt.weekday  # 0=Mon,6=Sun
    df["feat_is_month_start"] = dt.dt.is_month_start.astype(int)
    df["feat_is_month_end"]   = dt.dt.is_month_end.astype(int)

    # Collect all feature columns
    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    return df, feature_cols


def chrono_train_test_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: first 'train_ratio' fraction for training,
    remaining for test. Assumes df already sorted by draw_date.
    """
    n = len(df)
    if n < 50:
        raise ValueError(f"Not enough rows after cleaning for a stable split (got {n}, need ~50+).")

    split_idx = int(n * train_ratio)
    if split_idx <= 10 or n - split_idx <= 10:
        raise ValueError(
            f"Train/test split too extreme for dataset size {n}. "
            f"Try a train ratio between 0.6 and 0.9."
        )

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return train_df, test_df


def compute_binary_metrics(y_true, prob_1) -> dict:
    """
    For binary tasks:
    y_true: array-like of 0/1
    prob_1: predicted probability of class 1
    """
    y_true = np.asarray(y_true).astype(int)
    prob_1 = np.asarray(prob_1).astype(float)
    pred = (prob_1 >= 0.5).astype(int)

    acc = float((pred == y_true).mean())
    # Brier score
    brier = float(np.mean((prob_1 - y_true) ** 2))
    # Simple logloss
    eps = 1e-15
    logloss = float(
        -np.mean(y_true * np.log(prob_1 + eps) + (1 - y_true) * np.log(1 - prob_1 + eps))
    )
    return {"accuracy": acc, "brier": brier, "logloss": logloss}


def compute_multiclass_metrics(y_true, prob_matrix) -> dict:
    """
    For 4-class task:
    y_true: array-like of ints in {0,1,2,3}
    prob_matrix: shape (n_samples, 4)
    """
    y_true = np.asarray(y_true).astype(int)
    prob_matrix = np.asarray(prob_matrix).astype(float)
    pred = np.argmax(prob_matrix, axis=1)

    acc = float((pred == y_true).mean())
    eps = 1e-15
    # logloss = -mean(log p_true)
    true_probs = prob_matrix[np.arange(len(y_true)), y_true]
    logloss = float(-np.mean(np.log(true_probs + eps)))
    return {"accuracy": acc, "logloss": logloss}


def walk_forward_binary_backtest(
    df: pd.DataFrame,
    feat_cols: list,
    target_col: str,
    max_test_points: int = 80,
) -> tuple[dict, pd.DataFrame]:
    """
    Walk-forward expanding-window backtest for a binary target.
    - Train on rows [0 : idx), test on row idx, for a set of idx's
    - Only uses *past* rows for each prediction
    Limits to at most `max_test_points` test rows (latest ones).
    Returns:
        metrics dict, and a DataFrame with per-draw results.
    """
    n = len(df)
    if n < 80:
        raise ValueError(f"Need at least ~80 rows for a meaningful walk-forward test (got {n}).")

    # Start training after at least 50 rows
    min_train_size = max(50, int(n * 0.4))
    test_indices = list(range(min_train_size, n))

    if len(test_indices) > max_test_points:
        test_indices = test_indices[-max_test_points:]  # focus on most recent segment

    y_true = []
    prob1_list = []
    dates = []
    values = []

    for idx in test_indices:
        train_df = df.iloc[:idx]
        test_row = df.iloc[idx:idx + 1]

        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test  = test_row[feat_cols]

        model = CatBoostClassifier(
            loss_function="Logloss",
            depth=6,
            learning_rate=0.05,
            n_estimators=200,
            random_seed=100 + idx,
            verbose=False,
        )
        model.fit(X_train, y_train, verbose=False)
        prob1 = model.predict_proba(X_test)[0, 1]

        y_true.append(int(test_row[target_col].iloc[0]))
        prob1_list.append(float(prob1))
        dates.append(test_row["draw_date"].iloc[0])
        values.append(int(test_row["lottery_value"].iloc[0]))

    metrics = compute_binary_metrics(y_true, prob1_list)

    results_df = pd.DataFrame(
        {
            "draw_date": dates,
            "lottery_value": values,
            f"actual_{target_col}": y_true,
            f"pred_p1_{target_col}": np.round(prob1_list, 4),
        }
    )
    return metrics, results_df


def walk_forward_multiclass_backtest(
    df: pd.DataFrame,
    feat_cols: list,
    max_test_points: int = 60,
) -> tuple[dict, pd.DataFrame]:
    """
    Walk-forward expanding-window backtest for the 4-class target.
    - Train on rows [0 : idx), test on row idx
    Limits to at most `max_test_points` test rows (latest ones).
    Returns:
        metrics dict, and a DataFrame with per-draw results.
    """
    n = len(df)
    if n < 80:
        raise ValueError(f"Need at least ~80 rows for a meaningful walk-forward test (got {n}).")

    min_train_size = max(50, int(n * 0.4))
    test_indices = list(range(min_train_size, n))
    if len(test_indices) > max_test_points:
        test_indices = test_indices[-max_test_points:]

    y_true = []
    dates = []
    values = []
    all_probs = []

    for idx in test_indices:
        train_df = df.iloc[:idx]
        test_row = df.iloc[idx:idx + 1]

        X_train = train_df[feat_cols]
        y_train = train_df["target_4class"]
        X_test  = test_row[feat_cols]

        model = CatBoostClassifier(
            loss_function="MultiClass",
            depth=6,
            learning_rate=0.05,
            n_estimators=250,
            random_seed=200 + idx,
            verbose=False,
        )
        model.fit(X_train, y_train, verbose=False)
        prob_vec = model.predict_proba(X_test)[0]

        y_true.append(int(test_row["target_4class"].iloc[0]))
        dates.append(test_row["draw_date"].iloc[0])
        values.append(int(test_row["lottery_value"].iloc[0]))
        all_probs.append(prob_vec)

    prob_matrix = np.vstack(all_probs)
    metrics = compute_multiclass_metrics(y_true, prob_matrix)

    results_df = pd.DataFrame(
        {
            "draw_date": dates,
            "lottery_value": values,
            "actual_4class": y_true,
        }
    )
    for cls in range(4):
        results_df[f"p_cls_{cls}"] = np.round(prob_matrix[:, cls], 4)

    return metrics, results_df


def build_next_feature_row(
    df: pd.DataFrame,
    feat_cols: list,
    next_date=None,
) -> pd.DataFrame:
    """
    Build a single feature row for the *next* draw, based on all past draws.
    Uses the same logic as prepare_lottery_dataset_for_parity_range, but in 1-row form.
    """
    if len(df) == 0:
        raise ValueError("No rows available to build next-draw features.")

    df_sorted = df.sort_values("draw_date").reset_index(drop=True)
    dt_last = df_sorted["draw_date"].iloc[-1]

    if next_date is None:
        next_draw_date = dt_last + pd.Timedelta(days=1)
    else:
        next_draw_date = pd.to_datetime(next_date)

    parity = df_sorted["target_parity"].astype(int).values
    high   = df_sorted["target_range"].astype(int).values
    n = len(df_sorted)

    last_parity = parity[-1]
    last_high   = high[-1]

    # Previous flags for next row = last draw's actual class
    feat_odd_prev  = last_parity
    feat_high_prev = last_high

    # Rolling ratios over WINDOWS_PARITY_RANGE
    feat = {}
    feat["feat_odd_prev"]  = feat_odd_prev
    feat["feat_high_prev"] = feat_high_prev

    for w in WINDOWS_PARITY_RANGE:
        start_idx = max(0, n - w)
        feat[f"feat_odd_ratio_{w}"] = float(parity[start_idx:n].mean())
        feat[f"feat_high_ratio_{w}"] = float(high[start_idx:n].mean())

    # Streaks: use existing streaks to extend
    last_odd_streak_prev  = int(df_sorted["feat_odd_streak_prev"].iloc[-1])
    last_high_streak_prev = int(df_sorted["feat_high_streak_prev"].iloc[-1])

    if last_parity == 1:
        feat["feat_odd_streak_prev"] = last_odd_streak_prev + 1
    else:
        feat["feat_odd_streak_prev"] = 0

    if last_high == 1:
        feat["feat_high_streak_prev"] = last_high_streak_prev + 1
    else:
        feat["feat_high_streak_prev"] = 0

    # Calendar features
    feat["feat_year"]           = next_draw_date.year
    feat["feat_month"]          = next_draw_date.month
    feat["feat_day"]            = next_draw_date.day
    feat["feat_dow"]            = next_draw_date.weekday()
    feat["feat_is_month_start"] = int(next_draw_date.is_month_start)
    feat["feat_is_month_end"]   = int(next_draw_date.is_month_end)

    # Build DataFrame and order cols exactly like feat_cols
    row_df = pd.DataFrame([feat])
    # In case feat_cols include any extras (shouldn't), reindex with fill_value=0
    row_df = row_df.reindex(columns=feat_cols, fill_value=0)
    return row_df


FOUR_CLASS_LABELS = {
    0: "Even & Low (0–49 even)",
    1: "Odd & Low (0–49 odd)",
    2: "Even & High (50–99 even)",
    3: "Odd & High (50–99 odd)",
}

# ---------- Streamlit UI: New Section + Tabs ---------- #

st.markdown("---")
st.subheader("Parity / Range Prediction (per lottery) – Strategy B & C")

uploaded_cls = st.file_uploader(
    "Upload CSV for Parity/Range models (same lottery CSV as elsewhere)",
    type=["csv"],
    key="csv_parity_range",
)

if uploaded_cls is None:
    st.info("Upload a CSV file here to use the Odd/Even & Low/High models.")
else:
    try:
        df_cls_raw = load_csv_robust_for_classifiers(uploaded_cls)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df_cls_raw = None

    if df_cls_raw is not None:
        # Detect available lotteries (DR, FB, SG, GZ, GL, GB)
        lottery_cols = get_lottery_columns(df_cls_raw)
        if not lottery_cols:
            st.error(
                "Could not find any lottery columns (DR, FB, SG, GZ, GL, GB) "
                "in the uploaded CSV. Please check your headers."
            )
        else:
            st.success(f"Detected lottery columns: {', '.join(lottery_cols)}")

            # Global controls for both tabs
            chosen_lottery = st.selectbox(
                "Choose lottery column",
                options=lottery_cols,
                index=0,
                help="Models will be trained ONLY for this lottery (per-lottery, not ensemble).",
            )

            train_ratio = st.slider(
                "Train fraction (chronological split)",
                min_value=0.6,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="First fraction of rows (by date) used for training; remaining for test.",
            )

            if not CATBOOST_AVAILABLE:
                st.error(
                    "CatBoost is not available in this environment. "
                    "Run `pip install catboost` and restart the app to use these models."
                )
            else:
                # Prepare dataset once; reuse in both tabs.
                try:
                    df_cls, feat_cols = prepare_lottery_dataset_for_parity_range(
                        df_cls_raw, chosen_lottery
                    )
                except Exception as e:
                    st.error(f"Error preparing dataset for '{chosen_lottery}': {e}")
                    df_cls = None

                if df_cls is not None:
                    last_date = df_cls["draw_date"].max()
                    default_next_date = (last_date + timedelta(days=1)).date()

                    tab_bin, tab_four = st.tabs(
                        [
                            "Binary Parity & Range (Strategy B)",
                            "4-Class Parity+Range (Strategy C)",
                        ]
                    )

                    # ====== TAB 1: Strategy B – Two Binary Classifiers ====== #
                    with tab_bin:
                        st.markdown("### Strategy B – Dedicated Binary Classifiers")
                        st.write(
                            "Trains two separate CatBoost models, **per lottery**:\n"
                            "- Odd (1) vs Even (0)\n"
                            "- High (1, 50–99) vs Low (0, 0–49)\n"
                            "All features are based only on *past* draws for this lottery."
                        )

                        # --- Simple chronological split backtest --- #
                        if st.button(
                            f"Train & Evaluate (Chrono Split) – {chosen_lottery}",
                            key="btn_train_binary_parity_range",
                        ):
                            try:
                                train_df, test_df = chrono_train_test_split(df_cls, train_ratio)

                                X_train = train_df[feat_cols]
                                X_test  = test_df[feat_cols]

                                # ----- Parity model ----- #
                                y_parity_train = train_df["target_parity"]
                                y_parity_test  = test_df["target_parity"]

                                model_parity = CatBoostClassifier(
                                    loss_function="Logloss",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    random_seed=42,
                                    verbose=False,
                                )
                                model_parity.fit(X_train, y_parity_train, verbose=False)
                                prob_parity_test = model_parity.predict_proba(X_test)[:, 1]
                                metrics_parity = compute_binary_metrics(
                                    y_parity_test, prob_parity_test
                                )

                                # ----- Range model ----- #
                                y_range_train = train_df["target_range"]
                                y_range_test  = test_df["target_range"]

                                model_range = CatBoostClassifier(
                                    loss_function="Logloss",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    random_seed=43,
                                    verbose=False,
                                )
                                model_range.fit(X_train, y_range_train, verbose=False)
                                prob_range_test = model_range.predict_proba(X_test)[:, 1]
                                metrics_range = compute_binary_metrics(
                                    y_range_test, prob_range_test
                                )

                                st.markdown("#### Chronological Split – Parity (Odd vs Even)")
                                st.write(
                                    f"**Accuracy**: {metrics_parity['accuracy']:.3f}  \n"
                                    f"**Brier score**: {metrics_parity['brier']:.4f}  \n"
                                    f"**Logloss**: {metrics_parity['logloss']:.4f}"
                                )

                                y_par_pred = (prob_parity_test >= 0.5).astype(int)
                                cm_par = pd.crosstab(
                                    pd.Series(y_parity_test, name="Actual"),
                                    pd.Series(y_par_pred, name="Predicted"),
                                )
                                st.write("Confusion matrix – Parity:")
                                st.dataframe(cm_par)

                                st.markdown("---")
                                st.markdown("#### Chronological Split – Range (0–49 Low vs 50–99 High)")
                                st.write(
                                    f"**Accuracy**: {metrics_range['accuracy']:.3f}  \n"
                                    f"**Brier score**: {metrics_range['brier']:.4f}  \n"
                                    f"**Logloss**: {metrics_range['logloss']:.4f}"
                                )

                                y_rng_pred = (prob_range_test >= 0.5).astype(int)
                                cm_rng = pd.crosstab(
                                    pd.Series(y_range_test, name="Actual"),
                                    pd.Series(y_rng_pred, name="Predicted"),
                                )
                                st.write("Confusion matrix – Range:")
                                st.dataframe(cm_rng)

                                # Show last few test rows with predictions for inspection
                                st.markdown("#### Sample of test rows with predictions (Chrono Split)")
                                sample_show = test_df[["draw_date", "lottery_value"]].copy()
                                sample_show["Actual Parity (0=Even,1=Odd)"] = y_parity_test.values
                                sample_show["Pred P(odd)"] = np.round(prob_parity_test, 3)
                                sample_show["Actual Range (0=0–49,1=50–99)"] = y_range_test.values
                                sample_show["Pred P(high 50–99)"] = np.round(prob_range_test, 3)
                                st.dataframe(sample_show.tail(20))

                            except Exception as e:
                                st.error(f"Error training/evaluating binary models (chrono split): {e}")

                        st.markdown("----")
                        st.markdown("### Walk-Forward Backtest (Binary Models)")

                        max_walk_tests = st.slider(
                            "Max walk-forward test points (latest draws)",
                            min_value=30,
                            max_value=200,
                            value=80,
                            step=10,
                            key="max_walk_binary",
                        )

                        if st.button(
                            f"Run Walk-Forward Backtest – {chosen_lottery}",
                            key="btn_walk_forward_binary",
                        ):
                            try:
                                # Parity walk-forward
                                m_parity_wf, df_parity_wf = walk_forward_binary_backtest(
                                    df_cls, feat_cols, "target_parity", max_test_points=max_walk_tests
                                )
                                # Range walk-forward
                                m_range_wf, df_range_wf = walk_forward_binary_backtest(
                                    df_cls, feat_cols, "target_range", max_test_points=max_walk_tests
                                )

                                st.markdown("#### Walk-Forward – Parity (Odd vs Even)")
                                st.write(
                                    f"**Accuracy**: {m_parity_wf['accuracy']:.3f}  \n"
                                    f"**Brier score**: {m_parity_wf['brier']:.4f}  \n"
                                    f"**Logloss**: {m_parity_wf['logloss']:.4f}"
                                )
                                st.write("Last few walk-forward parity predictions:")
                                st.dataframe(df_parity_wf.tail(20))

                                st.markdown("---")
                                st.markdown("#### Walk-Forward – Range (0–49 Low vs 50–99 High)")
                                st.write(
                                    f"**Accuracy**: {m_range_wf['accuracy']:.3f}  \n"
                                    f"**Brier score**: {m_range_wf['brier']:.4f}  \n"
                                    f"**Logloss**: {m_range_wf['logloss']:.4f}"
                                )
                                st.write("Last few walk-forward range predictions:")
                                st.dataframe(df_range_wf.tail(20))

                            except Exception as e:
                                st.error(f"Error in walk-forward binary backtest: {e}")

                        st.markdown("----")
                        st.markdown("### Predict Next Draw (Binary Models)")

                        next_date_bin = st.date_input(
                            "Next draw date for prediction",
                            value=default_next_date,
                            key="next_date_binary",
                            help="Used for calendar features; history-based features use all past draws.",
                        )

                        if st.button(
                            f"Train on all past data & predict next draw – {chosen_lottery}",
                            key="btn_predict_next_binary",
                        ):
                            try:
                                X_all = df_cls[feat_cols]
                                y_par_all = df_cls["target_parity"]
                                y_rng_all = df_cls["target_range"]

                                # Train full-history models
                                model_par_full = CatBoostClassifier(
                                    loss_function="Logloss",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=350,
                                    random_seed=500,
                                    verbose=False,
                                )
                                model_par_full.fit(X_all, y_par_all, verbose=False)

                                model_rng_full = CatBoostClassifier(
                                    loss_function="Logloss",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=350,
                                    random_seed=501,
                                    verbose=False,
                                )
                                model_rng_full.fit(X_all, y_rng_all, verbose=False)

                                # Build next feature row
                                next_feat_row = build_next_feature_row(
                                    df_cls,
                                    feat_cols,
                                    next_date=next_date_bin,
                                )

                                p_odd = float(model_par_full.predict_proba(next_feat_row)[0, 1])
                                p_even = 1.0 - p_odd
                                p_high = float(model_rng_full.predict_proba(next_feat_row)[0, 1])
                                p_low  = 1.0 - p_high

                                pred_parity = "Odd" if p_odd >= 0.5 else "Even"
                                pred_range  = "High (50–99)" if p_high >= 0.5 else "Low (0–49)"

                                st.markdown("#### Next Draw Prediction – Binary Models")
                                st.write(f"**Lottery**: {chosen_lottery}")
                                st.write(f"**Next draw date**: {next_date_bin.isoformat()}")
                                st.write(
                                    f"- Parity → **{pred_parity}**  "
                                    f"(P(odd) = {p_odd:.3f}, P(even) = {p_even:.3f})"
                                )
                                st.write(
                                    f"- Range  → **{pred_range}**  "
                                    f"(P(high 50–99) = {p_high:.3f}, P(low 0–49) = {p_low:.3f})"
                                )

                            except Exception as e:
                                st.error(f"Error predicting next draw (binary models): {e}")

                    # ====== TAB 2: Strategy C – Single 4-Class Model ====== #
                    with tab_four:
                        st.markdown("### Strategy C – Single 4-Class Model")
                        st.write(
                            "Trains one **multi-class CatBoost** model (per lottery) with 4 classes:\n"
                            "- 0: Even & Low (0–49 even)\n"
                            "- 1: Odd & Low (0–49 odd)\n"
                            "- 2: Even & High (50–99 even)\n"
                            "- 3: Odd & High (50–99 odd)\n\n"
                            "You can recover Odd/Even or Low/High probabilities by summing relevant class probabilities."
                        )

                        # --- Simple chronological split backtest --- #
                        if st.button(
                            f"Train & Evaluate (Chrono Split) – 4-Class {chosen_lottery}",
                            key="btn_train_4class_parity_range",
                        ):
                            try:
                                train_df, test_df = chrono_train_test_split(df_cls, train_ratio)

                                X_train = train_df[feat_cols]
                                X_test  = test_df[feat_cols]
                                y4_train = train_df["target_4class"]
                                y4_test  = test_df["target_4class"]

                                model4 = CatBoostClassifier(
                                    loss_function="MultiClass",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=350,
                                    random_seed=44,
                                    verbose=False,
                                )
                                model4.fit(X_train, y4_train, verbose=False)
                                prob4_test = model4.predict_proba(X_test)
                                metrics4 = compute_multiclass_metrics(y4_test, prob4_test)

                                st.markdown("#### Chronological Split – 4-Class Parity+Range")
                                st.write(
                                    f"**Accuracy**: {metrics4['accuracy']:.3f}  \n"
                                    f"**Logloss**: {metrics4['logloss']:.4f}"
                                )

                                # Map numeric labels to text for confusion matrix
                                y4_pred = np.argmax(prob4_test, axis=1)

                                actual_labels = pd.Series(
                                    [FOUR_CLASS_LABELS[int(v)] for v in y4_test],
                                    name="Actual",
                                )
                                pred_labels = pd.Series(
                                    [FOUR_CLASS_LABELS[int(v)] for v in y4_pred],
                                    name="Predicted",
                                )
                                cm4 = pd.crosstab(actual_labels, pred_labels)
                                st.write("Confusion matrix – 4 classes:")
                                st.dataframe(cm4)

                                # Show sample of test rows with probabilities
                                st.markdown("#### Sample of test rows with 4-class probabilities (Chrono Split)")
                                sample = test_df[["draw_date", "lottery_value"]].copy()
                                for cls in range(4):
                                    sample[f"P({cls}: {FOUR_CLASS_LABELS[cls]})"] = np.round(
                                        prob4_test[:, cls], 3
                                    )
                                st.dataframe(sample.tail(20))

                            except Exception as e:
                                st.error(f"Error training/evaluating 4-class model (chrono split): {e}")

                        st.markdown("----")
                        st.markdown("### Walk-Forward Backtest (4-Class Model)")

                        max_walk_tests_4 = st.slider(
                            "Max walk-forward test points (latest draws) – 4-class",
                            min_value=30,
                            max_value=150,
                            value=60,
                            step=10,
                            key="max_walk_4class",
                        )

                        if st.button(
                            f"Run Walk-Forward Backtest – 4-Class {chosen_lottery}",
                            key="btn_walk_forward_4class",
                        ):
                            try:
                                metrics4_wf, df4_wf = walk_forward_multiclass_backtest(
                                    df_cls, feat_cols, max_test_points=max_walk_tests_4
                                )

                                st.markdown("#### Walk-Forward – 4-Class Parity+Range")
                                st.write(
                                    f"**Accuracy**: {metrics4_wf['accuracy']:.3f}  \n"
                                    f"**Logloss**: {metrics4_wf['logloss']:.4f}"
                                )
                                st.write("Last few walk-forward 4-class predictions:")
                                st.dataframe(df4_wf.tail(20))

                            except Exception as e:
                                st.error(f"Error in walk-forward 4-class backtest: {e}")

                        st.markdown("----")
                        st.markdown("### Predict Next Draw (4-Class Model)")

                        next_date_4 = st.date_input(
                            "Next draw date for prediction (4-class)",
                            value=default_next_date,
                            key="next_date_4class",
                        )

                        if st.button(
                            f"Train on all past data & predict next draw – 4-Class {chosen_lottery}",
                            key="btn_predict_next_4class",
                        ):
                            try:
                                X_all = df_cls[feat_cols]
                                y4_all = df_cls["target_4class"]

                                model4_full = CatBoostClassifier(
                                    loss_function="MultiClass",
                                    depth=6,
                                    learning_rate=0.05,
                                    n_estimators=400,
                                    random_seed=600,
                                    verbose=False,
                                )
                                model4_full.fit(X_all, y4_all, verbose=False)

                                next_feat_row_4 = build_next_feature_row(
                                    df_cls,
                                    feat_cols,
                                    next_date=next_date_4,
                                )

                                prob4_next = model4_full.predict_proba(next_feat_row_4)[0]

                                # Decode main class
                                cls_idx = int(np.argmax(prob4_next))
                                main_label = FOUR_CLASS_LABELS[cls_idx]

                                # Derive Odd/Even & Low/High from the 4-class distribution
                                p_even = float(prob4_next[0] + prob4_next[2])
                                p_odd  = float(prob4_next[1] + prob4_next[3])
                                p_low  = float(prob4_next[0] + prob4_next[1])
                                p_high = float(prob4_next[2] + prob4_next[3])

                                st.markdown("#### Next Draw Prediction – 4-Class Model")
                                st.write(f"**Lottery**: {chosen_lottery}")
                                st.write(f"**Next draw date**: {next_date_4.isoformat()}")
                                st.write(f"Most likely 4-class bucket: **{main_label}**")
                                st.write("Class probabilities:")
                                for k in range(4):
                                    st.write(
                                        f"- {k}: {FOUR_CLASS_LABELS[k]} → {prob4_next[k]:.3f}"
                                    )

                                st.markdown("Derived Odd/Even & Range probabilities:")
                                st.write(
                                    f"- Parity: P(odd) = {p_odd:.3f}, P(even) = {p_even:.3f}"
                                )
                                st.write(
                                    f"- Range:  P(low 0–49) = {p_low:.3f}, "
                                    f"P(high 50–99) = {p_high:.3f}"
                                )

                            except Exception as e:
                                st.error(f"Error predicting next draw (4-class model): {e}")

