
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
    """
    Build per-number hit timestamp lists (ms since epoch), sorted by time.
    """
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
    """
    Build leak-free per-(date, number) training dataset with recency features.
    For each (date, num) we only use hits strictly BEFORE that date.
    """
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
            # Leak-free recency stats for this (date, num)
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
    """
    Train CatBoost on per-(date, num) dataset, with:
    - leak-free recency features
    - class weighting for hits
    """
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

    # Class weighting to favour hits (positive class)
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

    Returns:
        best_model, best_config, history_df
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

    # Probability profiles (weekly/monthly) use *history* df, already trimmed by caller.
    use_recency_profiles = prob_mode in ("weekly", "monthly", "both")
    profiles = ens_build_number_profiles(df, lottery_key, recency_mode=use_recency_profiles)
    weekly_profile = profiles["weekly_profile"]
    monthly_profile = profiles["monthly_profile"]
    max_weekly = profiles["max_weekly"]
    max_monthly = profiles["max_monthly"]

    # Recency hit dates (for 'recent' mode) computed from history df only
    hit_dates = ens_build_hit_dates(df, lottery_key)

    # ML scores for this date
    dt = datetime(prediction_date.year, prediction_date.month, prediction_date.day)
    dow_index = (dt.weekday() + 1) % 7  # JS-style 0..6, Sun=0
    dd = dt.day

    x1 = dow_index / 6.0
    x2 = dd / 31.0
    feats = []
    for num in range(100):
        x3 = num / 99.0
        # Recency features (same formulation as training, but strictly using history df)
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

    # For recency-weighted mode, we now build a sharper recency profile:
    # - exponential half-life
    # - per-date max-normalisation
    # - alpha/beta weighting between "last hit" and "short-window frequency".
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
            # Normalised recency + short-window frequency
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
    2. **ML — Ensemble Number Prediction (0–99)** — CatBoost replacement for your TensorFlow ensemble,
       including:
       - **Probability Weight Strategy** (Weekly / Monthly / Both / Recency-weighted 3-month focus)
       - **ML / Probability blending** slider
       - **Core / Mid / Edge** tier sizing within Top N.
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
        print(f"Parsed {count_dates} unique dates from CSV.")
    except Exception as e:
        print(f"Error parsing CSV: {e}")

built_df = st.session_state.get("built_df", None)

# -------------------- Tabs -------------------- #
tab1, tab2 = st.tabs(["Range Wins + CatBoost (Any WIN)", "ML — Ensemble Number Prediction (0–99)"])

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

# ==================== TAB 2: ML Ensemble 0–99 ==================== #
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

        # Recency-specific advanced settings (UI defaults; training may override via auto-search config)
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

        # Enforce Core + Mid ≤ Top N
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

        # Manual training
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

        # Auto-search for best val accuracy (over train_ratio + recency params)
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
            # Use recency settings from the trained model config if available
            eff_rec_half = recency_half_life
            eff_rec_window = recency_window_days
            if ens_cfg is not None:
                eff_rec_half = float(ens_cfg.get("recency_half_life", eff_rec_half))
                eff_rec_window = float(ens_cfg.get("recency_window_days", eff_rec_window))

            # Default prediction date: day after last data date
            last_date = built_df["date"].max().date()
            default_pred_date = last_date + timedelta(days=1)
            pred_date = st.date_input(
                "Prediction date (for ranking 0–99)",
                value=default_pred_date,
                key="ens_pred_date",
            )

            if st.button("Score numbers (Core / Mid / Edge)", key="ens_score_btn"):
                # Use only history up to the day *before* prediction_date (leak-free)
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

        # 3) Backtest ensemble vs actual for a date range (leak-free)
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

            # Normalize to (start, end)
            if isinstance(eval_date_sel, tuple) and len(eval_date_sel) == 2:
                eval_start, eval_end = eval_date_sel
            else:
                eval_start = eval_end = eval_date_sel

            if eval_start > eval_end:
                st.error("Evaluation start date is after end date.")
            else:
                # Use recency settings from trained config if available
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

                        # Actual results on day d (using full df)
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

                        # History up to the day *before* d
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

# ==================== Global Date Range Lookup ==================== #

st.markdown("---")
st.subheader("Date Range Lookup — WIN status for your Range/Custom")

if built_df is None or built_df.empty:
    st.info("Upload a CSV and click **Analyze & Prepare Data** first to use the lookup.")
else:
    # Use the same range/custom currently stored
    rmin, rmax = st.session_state["range"]
    custom_set = st.session_state["custom_set"]
    custom_only = st.session_state["custom_only"]

    min_date = built_df["date"].min().date()
    max_date = built_df["date"].max().date()

    st.caption(
        f"Current settings → Range: `{rmin}-{rmax}`, "
        f"Custom: {sorted(list(custom_set)) or 'None'}, "
        f"Mode: {'Custom only' if custom_only else 'Range + Custom'}"
    )

    # Date range selector
    date_selection = st.date_input(
        "Select date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="lookup_range",
    )

    # Normalize selection: can be a date or a (start, end) tuple
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        start_date = end_date = date_selection

    if start_date > end_date:
        st.error("Start date is after end date. Please fix the range.")
    else:
        if st.button("Run Lookup", key="lookup_btn"):
            mask = (
                (built_df["date"].dt.date >= start_date)
                & (built_df["date"].dt.date <= end_date)
            )
            df_range = built_df.loc[mask].copy()

            if df_range.empty:
                st.warning("No records found in this date range.")
            else:
                lookup_table = pd.DataFrame(
                    {
                        "Date": df_range["date_key"],
                        "Day": df_range["dow_label"],
                        "DR": df_range["DR"].apply(format_nums),
                        "DR Status": np.where(df_range["DR_win"], "WIN", "NOT WIN"),
                        "FB": df_range["FB"].apply(format_nums),
                        "FB Status": np.where(df_range["FB_win"], "WIN", "NOT WIN"),
                        "GZ/GB": df_range["GZGB"].apply(format_nums),
                        "GZ/GB Status": np.where(df_range["GZGB_win"], "WIN", "NOT WIN"),
                        "GL": df_range["GL"].apply(format_nums),
                        "GL Status": np.where(df_range["GL_win"], "WIN", "NOT WIN"),
                        "Any WIN": np.where(df_range["any_win"], "WIN", "NOT WIN"),
                    }
                )

                wins_in_range = int(df_range["any_win"].sum())
                total_in_range = len(df_range)
                st.metric(
                    "WIN days in this range",
                    f"{wins_in_range} / {total_in_range}",
                )

                st.dataframe(lookup_table, use_container_width=True, hide_index=True)

                csv_bytes = lookup_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download lookup result as CSV",
                    data=csv_bytes,
                    file_name="date_range_lookup.csv",
                    mime="text/csv",
                )
