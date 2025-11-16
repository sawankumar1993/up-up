
import io
import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Least-Bet & ML Ranker — 0–99", layout="wide")

# ===================== Data loading (wide -> long) =====================
LOTTERY_COL_CANDIDATES = ["DR", "SG", "FB", "GZ", "GL"]

def _coerce_int_or_nan(x):
    try:
        return int(x)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def load_and_longify(file_bytes: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding_errors="ignore")
    cols = {c.lower(): c for c in df.columns}
    year_col  = cols.get("year")
    month_col = cols.get("month")
    day_col   = cols.get("day")
    if not year_col or not month_col or not day_col:
        maybe_date = next((c for c in df.columns if c.lower() in ("date","dt")), None)
        if maybe_date is None:
            raise ValueError("CSV must have Year/Month/Day OR a single Date column.")
        df["date"] = pd.to_datetime(df[maybe_date], errors="coerce", infer_datetime_format=True)
    else:
        df["date"] = pd.to_datetime(
            df[year_col].str.strip()+"-"+df[month_col].str.strip()+"-"+df[day_col].str.strip(),
            errors="coerce",
            format="%Y-%m-%d"
        )

    lotteries = [c for c in df.columns if c in LOTTERY_COL_CANDIDATES]
    if not lotteries:
        lotteries = [c for c in df.columns if c not in (year_col, month_col, day_col, "date")]

    long = df.melt(id_vars="date", value_vars=lotteries, var_name="lottery", value_name="draw_raw")
    long["draw"] = long["draw_raw"].map(_coerce_int_or_nan)
    long = long.dropna(subset=["date"]).copy()
    long["lottery"] = long["lottery"].astype(str)
    long = long[(~long["draw"].isna()) & (long["draw"]>=0) & (long["draw"]<=99)]
    long["draw"] = long["draw"].astype(int)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    return long.sort_values(["lottery", "date"]).reset_index(drop=True)

# ===================== Heuristic A/B/C =====================
def build_recent_frequency_table(df_lot: pd.DataFrame, end_date: pd.Timestamp, lookback_days: int = 120):
    start_date = end_date - pd.Timedelta(days=lookback_days)
    window = df_lot[(df_lot["date"] > start_date) & (df_lot["date"] <= end_date)]
    return window["draw"].value_counts().reindex(range(100), fill_value=0).sort_index()

def recency_penalty(df_lot: pd.DataFrame, end_date: pd.Timestamp, half_life_days: int = 30):
    last_seen = df_lot.groupby("draw")["date"].max()
    days_since = pd.Series(index=range(100), dtype=float)
    for n in range(100):
        last = last_seen.get(n, pd.NaT)
        days_since.loc[n] = 365.0 if pd.isna(last) else (end_date - last).days
    ln2 = math.log(2.0)
    penalty = np.exp(- (days_since / max(1, half_life_days)) * ln2)
    return (penalty - penalty.min()) / (penalty.max() - penalty.min() + 1e-9)

def crowd_appeal_features():
    liked = set()
    for n in range(100):
        s = f"{n:02d}"
        a, b = int(s[0]), int(s[1])
        if n % 10 == 0: liked.add(n)
        if n % 5 == 0: liked.add(n)
        if a == b: liked.add(n)
        if s in {"07","70","17","71","27","72"}: liked.add(n)
        if s in {"12","21","13","31","69","96"}: liked.add(n)
        if n in {0,1,99,98,97}: liked.add(n)
    return liked

LIKED_SET = crowd_appeal_features()

def anti_crowd_score_static():
    base = pd.Series(0.0, index=range(100))
    for n in range(100):
        base.loc[n] = 1.0 if n in LIKED_SET else 0.0
    blurred = base.rolling(3, min_periods=1, center=True).mean()
    return (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-9)

ANTI_CROWD = anti_crowd_score_static()

def strategy_A_cold(df_lot: pd.DataFrame, target_date: pd.Timestamp, lookback=120):
    freq = build_recent_frequency_table(df_lot, target_date - pd.Timedelta(days=1), lookback_days=lookback)
    rec = recency_penalty(df_lot, target_date - pd.Timedelta(days=1), half_life_days=30)
    f = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
    pop = 0.75 * f + 0.25 * rec
    return (-pop).sort_values(ascending=False)

def strategy_B_anti_crowd(df_lot: pd.DataFrame, target_date: pd.Timestamp, lookback=90):
    freq = build_recent_frequency_table(df_lot, target_date - pd.Timedelta(days=1), lookback_days=lookback)
    f = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
    pop = 0.65 * ANTI_CROWD + 0.35 * f
    return (-pop).sort_values(ascending=False)

def strategy_C_ensemble(df_lot: pd.DataFrame, target_date: pd.Timestamp):
    a = strategy_A_cold(df_lot, target_date)
    b = strategy_B_anti_crowd(df_lot, target_date)
    score = (0.6 * a.add(0, fill_value=0) + 0.4 * b.add(0, fill_value=0))
    return score.sort_values(ascending=False)

def top_k_from_score(score_series: pd.Series, k=25) -> pd.DataFrame:
    s = score_series.head(k).rename("score")
    top = s.reset_index()
    first_col = top.columns[0]
    if first_col != "number":
        top = top.rename(columns={first_col: "number"})
    top["rank"] = np.arange(1, len(top) + 1)
    return top[["rank", "number", "score"]]

# ===================== CatBoost Ranker (ML Mode) =====================
def _digit_feats(n: int):
    s = f"{n:02d}"; a, b = int(s[0]), int(s[1])
    return {
        "rep": int(a == b),
        "round10": int(n % 10 == 0),
        "round5": int(n % 5 == 0),
        "sevenish": int("7" in s),
        "mirror": int(s in {"12","21","13","31","69","96"}),
        "edge": int(n in {0,1,98,99}),
        "birthday": int(1 <= n <= 31),
        "monthish": int(1 <= n <= 12),
        "sum7": int(a + b == 7),
        "sum9": int(a + b == 9),
    }

def _time_feats_for_day(d: pd.Timestamp):
    return {
        "dow": d.dayofweek,
        "month": d.month,
        "woy": int(d.isocalendar().week),
        "dom": d.day,
        "is_month_start": int(d.day == 1),
        "is_month_end": int((d + pd.offsets.Day(1)).day == 1),
    }

def _history_counters(df_hist: pd.DataFrame, ref_day: pd.Timestamp, window_days: int):
    win = df_hist[(df_hist["date"] > ref_day - pd.Timedelta(days=window_days)) & (df_hist["date"] <= ref_day)]
    return win["draw"].value_counts().reindex(range(100), fill_value=0).sort_index()

def _days_since_last(df_hist: pd.DataFrame, ref_day: pd.Timestamp):
    last_seen = df_hist.groupby("draw")["date"].max()
    out = pd.Series(index=range(100), dtype=float)
    for n in range(100):
        last = last_seen.get(n, pd.NaT)
        out.loc[n] = 999.0 if pd.isna(last) else (ref_day - last).days
    return out

def build_feature_matrix_for_ref(df_lot: pd.DataFrame, ref_day: pd.Timestamp) -> pd.DataFrame:
    df_hist = df_lot[df_lot["date"] <= ref_day]
    pred_day = ref_day + pd.Timedelta(days=1)
    cnt7  = _history_counters(df_hist, ref_day, 7)
    cnt30 = _history_counters(df_hist, ref_day, 30)
    cnt90 = _history_counters(df_hist, ref_day, 90)
    dsl   = _days_since_last(df_hist, ref_day)
    tfe = _time_feats_for_day(pred_day)
    rows = []
    for n in range(100):
        feats = {"number": n, "cnt7": int(cnt7.loc[n]), "cnt30": int(cnt30.loc[n]),
                 "cnt90": int(cnt90.loc[n]), "days_since": float(dsl.loc[n])}
        feats.update(_digit_feats(n)); feats.update(tfe)
        rows.append(feats)
    X = pd.DataFrame(rows); X["days_since"] = X["days_since"].clip(0, 999)
    return X

@st.cache_resource(show_spinner=False)
def train_catboost_ranker(df_lot: pd.DataFrame, end_day: pd.Timestamp,
                          train_window_days: int = 540, min_warmup_days: int = 90,
                          val_frac: float = 0.2):
    try:
        from catboost import CatBoostRanker, Pool
    except Exception as e:
        return None, None, f"CatBoost not available: {e}"
    last_day = min(end_day, df_lot["date"].max())
    first_day = max(df_lot["date"].min() + pd.Timedelta(days=min_warmup_days), last_day - pd.Timedelta(days=train_window_days))
    days = pd.date_range(first_day, last_day - pd.Timedelta(days=1), freq="D")
    if len(days) < 30:
        return None, None, "Not enough days to train."
    X_list, y_list, gid_list = [], [], []
    for i, ref_day in enumerate(days):
        X_ref = build_feature_matrix_for_ref(df_lot, ref_day)
        next_day = ref_day + pd.Timedelta(days=1)
        y_num = df_lot.loc[df_lot["date"] == next_day, "draw"]
        if y_num.empty: continue
        actual = int(y_num.iloc[0])
        y_ref = (X_ref["number"].values == actual).astype(int)
        X_list.append(X_ref.drop(columns=["number"]))
        y_list.append(y_ref)
        gid_list.append(np.full(len(X_ref), i, dtype=int))
    if not X_list: return None, None, "No training rows."
    X = pd.concat(X_list, ignore_index=True); y = np.concatenate(y_list).astype(int); group_id = np.concatenate(gid_list).astype(int)
    feature_cols = X.columns.tolist()
    unique_g = np.unique(group_id)
    val_groups = set(unique_g[int((1 - val_frac) * len(unique_g)):])
    is_val = np.array([g in val_groups for g in group_id])
    from catboost import Pool
    train_pool = Pool(X[~is_val], y[~is_val], group_id=group_id[~is_val])
    val_pool   = Pool(X[ is_val], y[ is_val], group_id=group_id[ is_val])
    from catboost import CatBoostRanker
    model = CatBoostRanker(
        iterations=800, learning_rate=0.07, depth=6, l2_leaf_reg=3.0,
        loss_function="YetiRank", eval_metric="NDCG:top=25",
        random_seed=42, od_type="Iter", od_wait=60, thread_count=-1, verbose=False
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, feature_cols, None

def cb_scores_for_target(df_lot: pd.DataFrame, target_date: pd.Timestamp, model, feature_cols):
    try:
        from catboost import Pool
    except Exception as e:
        return None, f"CatBoost not available: {e}"
    ref_day = target_date - pd.Timedelta(days=1)
    X_pred = build_feature_matrix_for_ref(df_lot, ref_day)
    numbers = X_pred["number"].values
    Xp = X_pred.drop(columns=["number"]).reindex(columns=feature_cols, fill_value=0)
    pool = Pool(Xp)
    scores = model.predict(pool)
    s = pd.Series(scores, index=numbers, name="score")
    return s.sort_values(ascending=False), None

# ===================== LightGBM Ranker (NEW) =====================
def _group_sizes_for_mask(group_id: np.ndarray, mask: np.ndarray):
    """Compute group sizes for LightGBM given a boolean mask over rows, preserving order."""
    sizes = []
    prev = None
    cnt = 0
    for g, m in zip(group_id, mask):
        if not m:
            continue
        if prev is None:
            prev = g
            cnt = 1
        elif g == prev:
            cnt += 1
        else:
            sizes.append(cnt)
            prev = g
            cnt = 1
    if cnt > 0:
        sizes.append(cnt)
    return sizes

@st.cache_resource(show_spinner=False)
def train_lgbm_ranker(df_lot: pd.DataFrame, end_day: pd.Timestamp,
                      train_window_days: int = 540, min_warmup_days: int = 90,
                      val_frac: float = 0.2):
    try:
        import lightgbm as lgb
    except Exception as e:
        return None, None, f"LightGBM not available: {e}"
    last_day = min(end_day, df_lot["date"].max())
    first_day = max(df_lot["date"].min() + pd.Timedelta(days=min_warmup_days), last_day - pd.Timedelta(days=train_window_days))
    days = pd.date_range(first_day, last_day - pd.Timedelta(days=1), freq="D")
    if len(days) < 30:
        return None, None, "Not enough days to train."
    X_list, y_list, gid_list = [], [], []
    gid_counter = 0
    for ref_day in days:
        X_ref = build_feature_matrix_for_ref(df_lot, ref_day)
        next_day = ref_day + pd.Timedelta(days=1)
        y_num = df_lot.loc[df_lot["date"] == next_day, "draw"]
        if y_num.empty:
            continue
        actual = int(y_num.iloc[0])
        y_ref = (X_ref["number"].values == actual).astype(int)
        X_list.append(X_ref.drop(columns=["number"]))
        y_list.append(y_ref)
        gid_list.append(np.full(len(X_ref), gid_counter, dtype=int))
        gid_counter += 1
    if not X_list:
        return None, None, "No training rows."
    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list).astype(int)
    group_id = np.concatenate(gid_list).astype(int)

    feature_cols = X.columns.tolist()
    unique_g = np.unique(group_id)
    split_idx = int((1 - val_frac) * len(unique_g))
    train_groups = unique_g[:split_idx]
    val_groups = unique_g[split_idx:]
    is_train = np.isin(group_id, train_groups)
    is_val = np.isin(group_id, val_groups)

    # Group sizes for LightGBM
    group_train = _group_sizes_for_mask(group_id, is_train)
    group_val = _group_sizes_for_mask(group_id, is_val)

    import lightgbm as lgb
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X[is_train], y[is_train],
        group=group_train,
        eval_set=[(X[is_val], y[is_val])],
        eval_group=[group_val],
        eval_at=[25],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)]
    )
    return model, feature_cols, None

def lgbm_scores_for_target(df_lot: pd.DataFrame, target_date: pd.Timestamp, model, feature_cols):
    try:
        import lightgbm as lgb  # just to confirm availability
    except Exception as e:
        return None, f"LightGBM not available: {e}"
    ref_day = target_date - pd.Timedelta(days=1)
    X_pred = build_feature_matrix_for_ref(df_lot, ref_day)
    numbers = X_pred["number"].values
    Xp = X_pred.drop(columns=["number"]).reindex(columns=feature_cols, fill_value=0)
    scores = model.predict(Xp)
    s = pd.Series(scores, index=numbers, name="score")
    return s.sort_values(ascending=False), None

# ===================== Ensembles with Anti-Crowd =====================
def _minmax(s: pd.Series):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def _softmax(x: np.ndarray, tau: float = 1.0):
    z = (x - x.max()) / max(1e-9, tau)
    e = np.exp(z)
    return e / e.sum()

def ensemble_scores_ac_cb(df_lot: pd.DataFrame, target_date: pd.Timestamp,
                          cb_model, cb_cols,
                          blend_mode: str = "Weighted",
                          alpha: float = 0.7,
                          rrf_k0: int = 60, rrf_beta: float = 1.0,
                          softmax_tau: float = 1.0, prior_lambda: float = 1.0,
                          confidence_gate: bool = False, gate_threshold: float = 0.03, gate_shift: float = 0.1):
    s_cb, err = cb_scores_for_target(df_lot, target_date, cb_model, cb_cols)
    if err or s_cb is None:
        return None
    s_ac = strategy_B_anti_crowd(df_lot, target_date)

    if blend_mode == "Weighted":
        cb_n = _minmax(s_cb)
        ac_n = _minmax(s_ac)
        a = float(alpha)
        if confidence_gate:
            top2 = cb_n.sort_values(ascending=False).iloc[:2].values
            margin = float(top2[0] - top2[1]) if len(top2) == 2 else 0.0
            if margin < gate_threshold:
                a = max(0.05, min(0.95, a - gate_shift))
        combined = a * cb_n + (1.0 - a) * ac_n
        return combined.sort_values(ascending=False)
    elif blend_mode == "RRF":
        r_cb = s_cb.rank(ascending=False, method="min").astype(int)
        r_ac = s_ac.rank(ascending=False, method="min").astype(int)
        rrf = 1.0 / (rrf_k0 + r_cb) + rrf_beta * (1.0 / (rrf_k0 + r_ac))
        return rrf.sort_values(ascending=False)
    else:
        cb_vals = s_cb.values.astype(float)
        p_cb = _softmax(cb_vals, tau=softmax_tau)
        p_cb = pd.Series(p_cb, index=s_cb.index)
        ac_n = _minmax(s_ac)
        mult = np.exp(prior_lambda * (ac_n - 0.5))
        p_star = p_cb * mult
        p_star = p_star / (p_star.sum() + 1e-12)
        return p_star.sort_values(ascending=False)

def ensemble_scores_ac_lgb(df_lot: pd.DataFrame, target_date: pd.Timestamp,
                           lgbm_model, lgbm_cols,
                           blend_mode: str = "Weighted",
                           alpha: float = 0.7,
                           rrf_k0: int = 60, rrf_beta: float = 1.0,
                           softmax_tau: float = 1.0, prior_lambda: float = 1.0,
                           confidence_gate: bool = False, gate_threshold: float = 0.03, gate_shift: float = 0.1):
    s_ml, err = lgbm_scores_for_target(df_lot, target_date, lgbm_model, lgbm_cols)
    if err or s_ml is None:
        return None
    s_ac = strategy_B_anti_crowd(df_lot, target_date)

    if blend_mode == "Weighted":
        ml_n = _minmax(s_ml)
        ac_n = _minmax(s_ac)
        a = float(alpha)
        if confidence_gate:
            top2 = ml_n.sort_values(ascending=False).iloc[:2].values
            margin = float(top2[0] - top2[1]) if len(top2) == 2 else 0.0
            if margin < gate_threshold:
                a = max(0.05, min(0.95, a - gate_shift))
        combined = a * ml_n + (1.0 - a) * ac_n
        return combined.sort_values(ascending=False)
    elif blend_mode == "RRF":
        r_ml = s_ml.rank(ascending=False, method="min").astype(int)
        r_ac = s_ac.rank(ascending=False, method="min").astype(int)
        rrf = 1.0 / (rrf_k0 + r_ml) + rrf_beta * (1.0 / (rrf_k0 + r_ac))
        return rrf.sort_values(ascending=False)
    else:
        ml_vals = s_ml.values.astype(float)
        p_ml = _softmax(ml_vals, tau=softmax_tau)
        p_ml = pd.Series(p_ml, index=s_ml.index)
        ac_n = _minmax(s_ac)
        mult = np.exp(prior_lambda * (ac_n - 0.5))
        p_star = p_ml * mult
        p_star = p_star / (p_star.sum() + 1e-12)
        return p_star.sort_values(ascending=False)

# ===================== Backtests (incl. Ensembles) =====================
def backtest(df_lot: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, k: int, method: str,
             cb_model=None, cb_cols=None, lgbm_model=None, lgbm_cols=None,
             cb_train_days=540, lgb_train_days=540,
             ensemble_params: dict | None = None):
    df_window = df_lot[(df_lot["date"]>=start) & (df_lot["date"]<=end)].copy()
    days = sorted(df_window["date"].unique().tolist())
    hits, total = 0, 0

    # Prepare models lazily if requested
    if method.startswith("ML (CatBoost") or method.startswith("Ensemble (B + CatBoost)"):
        if (cb_model is None) or (cb_cols is None):
            cb_model, cb_cols, err = train_catboost_ranker(df_lot, end, train_window_days=cb_train_days)
            if err:
                return 0, 0, 0.0, err
    if method.startswith("ML (LightGBM") or method.startswith("Ensemble (B + LightGBM)"):
        if (lgbm_model is None) or (lgbm_cols is None):
            lgbm_model, lgbm_cols, err = train_lgbm_ranker(df_lot, end, train_window_days=lgb_train_days)
            if err:
                return 0, 0, 0.0, err

    for d in days:
        row = df_window[df_window["date"] == d]
        if row.empty: continue
        actual = int(row.iloc[0]["draw"])

        if method.startswith("Coldness"):
            score = strategy_A_cold(df_lot, d)
        elif method.startswith("Anti"):
            score = strategy_B_anti_crowd(df_lot, d)
        elif method.startswith("Ensemble (C)"):
            score = strategy_C_ensemble(df_lot, d)
        elif method.startswith("Ensemble (B + CatBoost)"):
            score = ensemble_scores_ac_cb(df_lot, d, cb_model, cb_cols, **(ensemble_params or {}))
            if score is None: continue
        elif method.startswith("Ensemble (B + LightGBM)"):
            score = ensemble_scores_ac_lgb(df_lot, d, lgbm_model, lgbm_cols, **(ensemble_params or {}))
            if score is None: continue
        elif method.startswith("ML (CatBoost"):
            score, err = cb_scores_for_target(df_lot, d, cb_model, cb_cols)
            if err or score is None: continue
        elif method.startswith("ML (LightGBM"):
            score, err = lgbm_scores_for_target(df_lot, d, lgbm_model, lgbm_cols)
            if err or score is None: continue
        else:
            continue

        picks = set(score.head(k).index.tolist())
        hits += 1 if actual in picks else 0
        total += 1

    hitrate = (hits / total * 100.0) if total else 0.0
    return hits, total, hitrate, None

def compare_backtests(df_lot: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, k: int,
                      cb_train_days=540, lgb_train_days=540, ensemble_params: dict | None = None):
    methods = [
        "Coldness (A)", "Anti-Crowd (B)", "Ensemble (C)",
        "ML (CatBoost Ranker)", "Ensemble (B + CatBoost)",
        "ML (LightGBM Ranker)", "Ensemble (B + LightGBM)"
    ]
    results = []
    cb_model, cb_cols, _ = train_catboost_ranker(df_lot, end, train_window_days=cb_train_days)
    lgbm_model, lgbm_cols, _ = train_lgbm_ranker(df_lot, end, train_window_days=lgb_train_days)
    for m in methods:
        hits, total, hitrate, _err = backtest(
            df_lot, start, end, k, m,
            cb_model=cb_model, cb_cols=cb_cols,
            lgbm_model=lgbm_model, lgbm_cols=lgbm_cols,
            cb_train_days=cb_train_days, lgb_train_days=lgb_train_days,
            ensemble_params=ensemble_params
        )
        results.append({"Method": m, "Hits": hits, "Total": total, "Hit-rate %": round(hitrate, 1)})
    return pd.DataFrame(results)

# ===================== Helpers & UI =====================
def _fmt_date_for_metric(ts):
    if ts is None or pd.isna(ts): return None
    try:
        d = getattr(ts, "date", lambda: ts)()
        return str(d)
    except Exception:
        return str(ts)

st.title("🎯 Least-Bet Proxies, CatBoost / LightGBM Rankers & Ensembles — DR / SG / FB / GZ / GL")

with st.sidebar:
    st.header("1) Upload CSV")
    upl = st.file_uploader("CSV: Year, Month, Day, DR, SG, FB, GZ, GL", type=["csv"])
    st.caption("We treat 'xx' as missing and drop those rows for the given lottery/date.")

    st.header("2) Choose Lottery & Date")
    default_date = date.today()
    pick_date = st.date_input("Target date", value=default_date)

    method = st.radio(
        "Scoring method",
        [
            "Ensemble (C)",
            "Coldness (A)",
            "Anti-Crowd (B)",
            "ML (CatBoost Ranker)",
            "Ensemble (B + CatBoost)",
            "ML (LightGBM Ranker)",
            "Ensemble (B + LightGBM)",
        ],
        index=0,
        help="New: Choose LightGBM or CatBoost ML ranker, or blend them with Anti-Crowd."
    )
    k = st.slider("How many picks?", min_value=10, max_value=50, value=25, step=1)

    st.header("3) Backtest (optional)")
    do_bt = st.checkbox("Run a backtest")
    bt_days = st.slider("Backtest window (days before target)", min_value=30, max_value=365, value=120, step=10)

    st.header("4) ML settings")
    cb_train_days = st.slider("CatBoost training window (days)", min_value=120, max_value=900, value=540, step=30)
    lgb_train_days = st.slider("LightGBM training window (days)", min_value=120, max_value=900, value=540, step=30)

    st.header("5) Ensemble settings (for B + ML blends)")
    blend_mode = st.selectbox("Blend mode", ["Weighted", "RRF", "Prior-Adjusted"], index=0)
    alpha = st.slider("Weighted α (ML weight)", 0.05, 0.95, 0.70, 0.05)
    confidence_gate = st.checkbox("Confidence-aware blending", value=True)
    gate_threshold = st.slider("Gate threshold (ML top1–top2 gap)", 0.0, 0.2, 0.03, 0.01)
    gate_shift = st.slider("Gate shift (reduce α by)", 0.0, 0.3, 0.10, 0.05)
    rrf_k0 = st.slider("RRF k0", 10, 100, 60, 5)
    rrf_beta = st.slider("RRF β (Anti-Crowd weight)", 0.1, 2.0, 1.0, 0.1)
    softmax_tau = st.slider("Prior-Adjusted: temperature τ", 0.5, 2.0, 1.0, 0.1)
    prior_lambda = st.slider("Prior-Adjusted: prior strength λ", 0.1, 3.0, 1.0, 0.1)
    auto_tune_alpha = st.checkbox("Auto-tune α on backtest window (Weighted mode)", value=False)

if upl is None:
    st.info("Upload your CSV to begin.")
    st.stop()

# Load & prep
try:
    data = load_and_longify(upl.getvalue(), upl.name)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success(f"Loaded {len(data):,} rows, {data['date'].min().date()} → {data['date'].max().date()}.")

lotteries = sorted(data["lottery"].unique().tolist())
lottery = st.selectbox("Lottery", lotteries, index=0)

df_lot = data[data["lottery"] == lottery].copy()
if df_lot.empty:
    st.warning("No rows for this lottery.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Lottery", lottery)
with m2: st.metric("First date", _fmt_date_for_metric(df_lot["date"].min()))
with m3: st.metric("Last date", _fmt_date_for_metric(df_lot["date"].max()))
with m4: st.metric("Rows", int(len(df_lot)))

st.divider()

# Train models lazily only if needed
cb_model = cb_cols = None
lgbm_model = lgbm_cols = None

need_cb = method.startswith("ML (CatBoost") or method.startswith("Ensemble (B + CatBoost)")
need_lgb = method.startswith("ML (LightGBM") or method.startswith("Ensemble (B + LightGBM)")

if need_cb:
    with st.spinner("Training CatBoost Ranker (cached)…"):
        cb_model, cb_cols, err = train_catboost_ranker(df_lot, pd.to_datetime(pick_date) - pd.Timedelta(days=1), train_window_days=cb_train_days)
        if err:
            st.error(err + " — run: pip install catboost")
            st.stop()
if need_lgb:
    with st.spinner("Training LightGBM Ranker (cached)…"):
        lgbm_model, lgbm_cols, err = train_lgbm_ranker(df_lot, pd.to_datetime(pick_date) - pd.Timedelta(days=1), train_window_days=lgb_train_days)
        if err:
            st.error(err + " — run: pip install lightgbm")
            st.stop()

# Optional auto-tune α for B+ML weighted mode
if do_bt and blend_mode == "Weighted" and auto_tune_alpha and (method.startswith("Ensemble (B +")):
    end_bt = min(pd.to_datetime(pick_date), df_lot["date"].max())
    start_bt = end_bt - pd.Timedelta(days=bt_days)
    alphas = [0.5, 0.6, 0.7, 0.8]
    best = {"alpha": alpha, "hr": -1.0}
    for a in alphas:
        eparams = dict(blend_mode="Weighted", alpha=a, confidence_gate=confidence_gate,
                       gate_threshold=gate_threshold, gate_shift=gate_shift,
                       rrf_k0=rrf_k0, rrf_beta=rrf_beta,
                       softmax_tau=softmax_tau, prior_lambda=prior_lambda)
        if method.startswith("Ensemble (B + CatBoost)"):
            hits, total, hr, err = backtest(df_lot, start_bt, end_bt, k, "Ensemble (B + CatBoost)",
                                            cb_model=cb_model, cb_cols=cb_cols, cb_train_days=cb_train_days,
                                            ensemble_params=eparams)
        else:
            hits, total, hr, err = backtest(df_lot, start_bt, end_bt, k, "Ensemble (B + LightGBM)",
                                            lgbm_model=lgbm_model, lgbm_cols=lgbm_cols, lgb_train_days=lgb_train_days,
                                            ensemble_params=eparams)
        if hr > best["hr"]:
            best = {"alpha": a, "hr": hr}
    alpha = best["alpha"]
    st.info(f"Auto-tuned α = **{alpha:.2f}** on backtest window (Hit-rate {best['hr']:.1f}%).")

# Compute picks
tdate = pd.to_datetime(pick_date)
with st.spinner("Scoring numbers 0–99…"):
    if method.startswith("Coldness"):
        scores = strategy_A_cold(df_lot, tdate)
    elif method.startswith("Anti"):
        scores = strategy_B_anti_crowd(df_lot, tdate)
    elif method.startswith("Ensemble (C)"):
        scores = strategy_C_ensemble(df_lot, tdate)
    elif method.startswith("Ensemble (B + CatBoost)"):
        eparams = dict(
            blend_mode=blend_mode, alpha=alpha,
            rrf_k0=rrf_k0, rrf_beta=rrf_beta,
            softmax_tau=softmax_tau, prior_lambda=prior_lambda,
            confidence_gate=confidence_gate, gate_threshold=gate_threshold, gate_shift=gate_shift
        )
        scores = ensemble_scores_ac_cb(df_lot, tdate, cb_model, cb_cols, **eparams)
        if scores is None:
            st.error("CatBoost scoring failed — ensure catboost is installed.")
            st.stop()
    elif method.startswith("Ensemble (B + LightGBM)"):
        eparams = dict(
            blend_mode=blend_mode, alpha=alpha,
            rrf_k0=rrf_k0, rrf_beta=rrf_beta,
            softmax_tau=softmax_tau, prior_lambda=prior_lambda,
            confidence_gate=confidence_gate, gate_threshold=gate_threshold, gate_shift=gate_shift
        )
        scores = ensemble_scores_ac_lgb(df_lot, tdate, lgbm_model, lgbm_cols, **eparams)
        if scores is None:
            st.error("LightGBM scoring failed — ensure lightgbm is installed.")
            st.stop()
    elif method.startswith("ML (CatBoost"):
        scores, err = cb_scores_for_target(df_lot, tdate, cb_model, cb_cols)
        if err or scores is None:
            st.error(err or "CatBoost scoring failed.")
            st.stop()
    else:  # ML (LightGBM Ranker)
        scores, err = lgbm_scores_for_target(df_lot, tdate, lgbm_model, lgbm_cols)
        if err or scores is None:
            st.error(err or "LightGBM scoring failed.")
            st.stop()

results = top_k_from_score(scores, k=k)
st.subheader(f"Top-{k} picks for **{lottery}** on **{tdate.date()}** — Method: {method}")
st.dataframe(results, hide_index=True, use_container_width=True)
st.bar_chart(results.set_index("number")["score"])

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download picks as CSV",
    data=csv_bytes,
    file_name=f"picks_{lottery}_{tdate.date()}_{method.replace(' ','_')}_top{k}.csv",
    mime="text/csv",
    use_container_width=True
)

# Historical actual (Y/N quick look)
actual = df_lot[df_lot["date"] == tdate]["draw"]
if not actual.empty:
    actual_num = int(actual.iloc[0])
    in_topk = actual_num in set(results["number"])
    st.info(f"Actual drawn number for {lottery} on {tdate.date()}: **{actual_num:02d}** — "
            f"{'✅ In Top-'+str(k) if in_topk else '❌ Not in Top-'+str(k)}")

# Backtest (incl. Ensembles comparison)
if do_bt:
    end = min(tdate, df_lot["date"].max())
    start = end - pd.Timedelta(days=bt_days)
    st.subheader("Backtest Results")
    st.write(f"Window: **{start.date()} → {end.date()}** | Days with draws: **{df_lot[(df_lot['date']>=start) & (df_lot['date']<=end)].date.nunique()}**")

    if method.startswith("ML (CatBoost"):
        mdl, cols, err = train_catboost_ranker(df_lot, end, train_window_days=cb_train_days)
        hits, total, hitrate, _ = backtest(df_lot, start, end, k, method,
                                           cb_model=mdl, cb_cols=cols, cb_train_days=cb_train_days)
    elif method.startswith("Ensemble (B + CatBoost)"):
        eparams = dict(
            blend_mode=blend_mode, alpha=alpha,
            rrf_k0=rrf_k0, rrf_beta=rrf_beta,
            softmax_tau=softmax_tau, prior_lambda=prior_lambda,
            confidence_gate=confidence_gate, gate_threshold=gate_threshold, gate_shift=gate_shift
        )
        hits, total, hitrate, _ = backtest(df_lot, start, end, k, method,
                                           cb_model=cb_model, cb_cols=cb_cols, cb_train_days=cb_train_days,
                                           ensemble_params=eparams)
    elif method.startswith("ML (LightGBM"):
        mdl, cols, err = train_lgbm_ranker(df_lot, end, train_window_days=lgb_train_days)
        hits, total, hitrate, _ = backtest(df_lot, start, end, k, method,
                                           lgbm_model=mdl, lgbm_cols=cols, lgb_train_days=lgb_train_days)
    elif method.startswith("Ensemble (B + LightGBM)"):
        eparams = dict(
            blend_mode=blend_mode, alpha=alpha,
            rrf_k0=rrf_k0, rrf_beta=rrf_beta,
            softmax_tau=softmax_tau, prior_lambda=prior_lambda,
            confidence_gate=confidence_gate, gate_threshold=gate_threshold, gate_shift=gate_shift
        )
        hits, total, hitrate, _ = backtest(df_lot, start, end, k, method,
                                           lgbm_model=lgbm_model, lgbm_cols=lgbm_cols, lgb_train_days=lgb_train_days,
                                           ensemble_params=eparams)
    else:
        hits, total, hitrate, _ = backtest(df_lot, start, end, k, method,
                                           cb_train_days=cb_train_days, lgb_train_days=lgb_train_days)

    st.write(f"**{method}** → Hit-rate: **{hitrate:.1f}%** ({hits}/{total})")

    comp_df = compare_backtests(
        df_lot, start, end, k,
        cb_train_days=cb_train_days, lgb_train_days=lgb_train_days,
        ensemble_params=dict(
            blend_mode=blend_mode, alpha=alpha,
            rrf_k0=rrf_k0, rrf_beta=rrf_beta,
            softmax_tau=softmax_tau, prior_lambda=prior_lambda,
            confidence_gate=confidence_gate, gate_threshold=gate_threshold, gate_shift=gate_shift
        )
    )
    st.markdown("### Comparison of all methods")
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.bar_chart(comp_df.set_index("Method")["Hit-rate %"])


# ===================== Copy (ascending, one-per-line) — SAFE =====================
try:
    import json
    from streamlit.components.v1 import html as st_html

    nums_sorted = sorted(results["number"].astype(int).tolist())
    one_per_line = "\n".join(str(n) for n in nums_sorted)

    st.divider()
    st.subheader("📋 Copy your Top-K numbers (ascending, one per line)")

    payload = json.dumps(one_per_line)
    html_block = """
<div style="width:100%;">
  <textarea id="copyTA" rows="12" style="width:100%;font-family:monospace;" readonly></textarea>
  <div style="margin-top:6px;">
    <button id="copyBtn" style="padding:6px 10px; border-radius:6px; cursor:pointer;">Copy</button>
    <span id="copied" style="margin-left:8px;color:green;display:none;">Copied!</span>
  </div>
</div>
<script>
  const txt = __PAYLOAD__;
  const ta = document.getElementById('copyTA');
  ta.value = txt;
  document.getElementById('copyBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(ta.value);
    } catch (e) {
      ta.focus(); ta.select(); document.execCommand('copy');
    }
    const note = document.getElementById('copied');
    note.style.display = 'inline';
    setTimeout(() => note.style.display = 'none', 1200);
  });
</script>
""".replace("__PAYLOAD__", payload)

    st_html(html_block, height=260)

except Exception:
    try:
        st.text_area("Copy your Top-K numbers (ascending, one per line)", value=one_per_line, height=120)
    except Exception:
        pass


# ===================== Monthly Y/N coverage — SAFE =====================
try:
    import calendar, json
    from streamlit.components.v1 import html as st_html

    st.divider()
    st.subheader("🗓️ Monthly Top-K coverage (Y/N)")

    _base_date = pd.to_datetime(pick_date) if "pick_date" in locals() else pd.Timestamp.today()
    _year = st.number_input("Year (monthly check)", min_value=2000, max_value=2100, value=int(_base_date.year), step=1)
    _month = st.selectbox(
        "Month (monthly check)", list(range(1, 13)),
        index=int(_base_date.month) - 1,
        format_func=lambda m: calendar.month_name[m]
    )

    _start_month = pd.Timestamp(int(_year), int(_month), 1)
    _end_month = _start_month + pd.offsets.MonthEnd(1)

    _df_month = df_lot[(df_lot["date"] >= _start_month) & (df_lot["date"] <= _end_month)].copy()
    _days = sorted(_df_month["date"].unique().tolist())

    if not _days:
        st.info("No draws for this lottery in the selected month.")
    else:
        # Ensure model if needed
        if method.startswith("ML (CatBoost") or method.startswith("Ensemble (B + CatBoost)"):
            if (cb_model is None) or (cb_cols is None):
                with st.spinner("Training CatBoost for monthly coverage…"):
                    _train_days = cb_train_days if "cb_train_days" in locals() else 540
                    cb_model, cb_cols, _err = train_catboost_ranker(df_lot, _end_month, train_window_days=_train_days)
                    if _err:
                        st.error(_err + " — run: pip install catboost")
                        st.stop()
        if method.startswith("ML (LightGBM") or method.startswith("Ensemble (B + LightGBM)"):
            if (lgbm_model is None) or (lgbm_cols is None):
                with st.spinner("Training LightGBM for monthly coverage…"):
                    _train_days = lgb_train_days if "lgb_train_days" in locals() else 540
                    lgbm_model, lgbm_cols, _err = train_lgbm_ranker(df_lot, _end_month, train_window_days=_train_days)
                    if _err:
                        st.error(_err + " — run: pip install lightgbm")
                        st.stop()

        _eparams = dict(
            blend_mode=blend_mode if "blend_mode" in locals() else "Weighted",
            alpha=alpha if "alpha" in locals() else 0.7,
            rrf_k0=rrf_k0 if "rrf_k0" in locals() else 60,
            rrf_beta=rrf_beta if "rrf_beta" in locals() else 1.0,
            softmax_tau=softmax_tau if "softmax_tau" in locals() else 1.0,
            prior_lambda=prior_lambda if "prior_lambda" in locals() else 1.0,
            confidence_gate=confidence_gate if "confidence_gate" in locals() else True,
            gate_threshold=gate_threshold if "gate_threshold" in locals() else 0.03,
            gate_shift=gate_shift if "gate_shift" in locals() else 0.10,
        )

        _yn_lines = []
        _rows_html = []
        for _d in _days:
            _actual = int(_df_month.loc[_df_month["date"] == _d, "draw"].iloc[0])
            _score = None
            if method.startswith("Coldness"):
                _score = strategy_A_cold(df_lot, _d)
            elif method.startswith("Anti"):
                _score = strategy_B_anti_crowd(df_lot, _d)
            elif method.startswith("Ensemble (C)"):
                _score = strategy_C_ensemble(df_lot, _d)
            elif method.startswith("Ensemble (B + CatBoost)"):
                _score = ensemble_scores_ac_cb(df_lot, _d, cb_model, cb_cols, **_eparams)
            elif method.startswith("Ensemble (B + LightGBM)"):
                _score = ensemble_scores_ac_lgb(df_lot, _d, lgbm_model, lgbm_cols, **_eparams)
            elif method.startswith("ML (CatBoost"):
                _score, _err2 = cb_scores_for_target(df_lot, _d, cb_model, cb_cols)
            else:
                _score, _err2 = lgbm_scores_for_target(df_lot, _d, lgbm_model, lgbm_cols)

            if _score is None:
                _yn = "N"
            else:
                _picks = set(_score.head(k).index.tolist())
                _yn = "Y" if _actual in _picks else "N"

            _yn_lines.append(_yn)

            _bg = "#d4edda" if _yn == "Y" else "#f8d7da"
            _fg = "#155724" if _yn == "Y" else "#721c24"
            _rows_html.append(
                f"<tr><td class='dt'>{pd.to_datetime(_d).date()}</td>"
                f"<td class='yn' style='background:{_bg};color:{_fg};text-align:center;font-weight:600;'>{_yn}</td></tr>"
            )

        _table_html = (
            "<table id='ynTable'><thead><tr><th class='dt'>Date</th><th>In Top-K</th></tr></thead><tbody>"
            + "".join(_rows_html) + "</tbody></table>"
        )

        table_payload = json.dumps(_table_html)
        plain_payload = json.dumps("\n".join(_yn_lines))

        html_month = """
<style>
#ynTable {
  border-collapse: collapse;
  font-family: monospace;
  width: 100%;
}
#ynTable th, #ynTable td {
  border: 1px solid #ddd;
  padding: 6px 10px;
}
#ynTable th.dt, #ynTable td.dt {
  background: #333333;
  color: #ffffff;
}
#ynScrollable {
  max-height: 520px;
  overflow: auto;
  border: 1px solid #eee;
  border-radius: 6px;
  padding: 6px;
  background: #fff;
}
</style>

<div style="margin:6px 0;">
  <div id="ynScrollable">__TABLE__</div>
  <div style="margin-top:8px;">
    <button id="copyHtmlBtn" style="padding:6px 10px; border-radius:6px; cursor:pointer;">Copy colored table</button>
    <button id="copyPlainBtn" style="padding:6px 10px; border-radius:6px; cursor:pointer; margin-left:6px;">Copy Y/N (plain)</button>
    <span id="copiedMsg" style="margin-left:8px;color:green;display:none;">Copied!</span>
  </div>
</div>

<script>
  const htmlTable = __TABLE_JSON__;
  const plainText = __PLAIN_JSON__;

  document.getElementById('ynScrollable').innerHTML = htmlTable;

  async function copyHTMLWithFallback(html, plain) {
    if (navigator.clipboard && window.ClipboardItem) {
      try {
        const item = new ClipboardItem({
          'text/html': new Blob([html], {type: 'text/html'}),
          'text/plain': new Blob([plain], {type: 'text/plain'}),
        });
        await navigator.clipboard.write([item]);
        return true;
      } catch (e) {}
    }
    try {
      const hidden = document.createElement('div');
      hidden.contentEditable = 'true';
      hidden.style.position = 'fixed';
      hidden.style.left = '-9999px';
      hidden.innerHTML = html;
      document.body.appendChild(hidden);
      const range = document.createRange();
      range.selectNodeContents(hidden);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand('copy');
      document.body.removeChild(hidden);
      return true;
    } catch (e) {
      return false;
    }
  }

  document.getElementById('copyHtmlBtn').addEventListener('click', async () => {
    const ok = await copyHTMLWithFallback(htmlTable, plainText);
    const note = document.getElementById('copiedMsg');
    note.textContent = ok ? 'Copied colored table!' : 'Copy failed';
    note.style.display = 'inline';
    setTimeout(() => note.style.display = 'none', 1500);
  });

  document.getElementById('copyPlainBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(plainText);
    } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = plainText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    const note = document.getElementById('copiedMsg');
    note.textContent = 'Copied plain Y/N!';
    note.style.display = 'inline';
    setTimeout(() => note.style.display = 'none', 1500);
  });
</script>
""".replace("__TABLE__", _table_html).replace("__TABLE_JSON__", table_payload).replace("__PLAIN_JSON__", plain_payload)

        st_html(html_month, height=640)

except Exception:
    pass


# ===================== YEARLY Top-K coverage (on demand) — SAFE =====================
try:
    import json, calendar
    from streamlit.components.v1 import html as st_html

    st.divider()
    st.subheader("📆 Yearly Top-K coverage (Y/N)")

    _base_date_y = pd.to_datetime(pick_date) if "pick_date" in locals() else pd.Timestamp.today()
    _year_y = st.number_input("Year (yearly check)", min_value=2000, max_value=2100, value=int(_base_date_y.year), step=1)
    _do_year = st.checkbox("Calculate for the selected year", value=False)

    if _do_year:
        _start_year = pd.Timestamp(int(_year_y), 1, 1)
        _end_year   = pd.Timestamp(int(_year_y), 12, 31)

        _df_year = df_lot[(df_lot["date"] >= _start_year) & (df_lot["date"] <= _end_year)].copy()
        _ydays = sorted(_df_year["date"].unique().tolist())

        if not _ydays:
            st.info("No draws for this lottery in the selected year.")
        else:
            # Ensure model if needed
            if method.startswith("ML (CatBoost") or method.startswith("Ensemble (B + CatBoost)"):
                if (cb_model is None) or (cb_cols is None):
                    with st.spinner("Training CatBoost for yearly coverage…"):
                        _train_days_y = cb_train_days if "cb_train_days" in locals() else 540
                        cb_model, cb_cols, _err_y = train_catboost_ranker(df_lot, _end_year, train_window_days=_train_days_y)
                        if _err_y:
                            st.error(_err_y + " — run: pip install catboost")
                            st.stop()
            if method.startswith("ML (LightGBM") or method.startswith("Ensemble (B + LightGBM)"):
                if (lgbm_model is None) or (lgbm_cols is None):
                    with st.spinner("Training LightGBM for yearly coverage…"):
                        _train_days_y = lgb_train_days if "lgb_train_days" in locals() else 540
                        lgbm_model, lgbm_cols, _err_y = train_lgbm_ranker(df_lot, _end_year, train_window_days=_train_days_y)
                        if _err_y:
                            st.error(_err_y + " — run: pip install lightgbm")
                            st.stop()

            _eparams_y = dict(
                blend_mode=blend_mode if "blend_mode" in locals() else "Weighted",
                alpha=alpha if "alpha" in locals() else 0.7,
                rrf_k0=rrf_k0 if "rrf_k0" in locals() else 60,
                rrf_beta=rrf_beta if "rrf_beta" in locals() else 1.0,
                softmax_tau=softmax_tau if "softmax_tau" in locals() else 1.0,
                prior_lambda=prior_lambda if "prior_lambda" in locals() else 1.0,
                confidence_gate=confidence_gate if "confidence_gate" in locals() else True,
                gate_threshold=gate_threshold if "gate_threshold" in locals() else 0.03,
                gate_shift=gate_shift if "gate_shift" in locals() else 0.10,
            )

            _yn_year_lines = []
            _rows_year_html = []
            for _d in _ydays:
                _actual = int(_df_year.loc[_df_year["date"] == _d, "draw"].iloc[0])

                _score = None
                if method.startswith("Coldness"):
                    _score = strategy_A_cold(df_lot, _d)
                elif method.startswith("Anti"):
                    _score = strategy_B_anti_crowd(df_lot, _d)
                elif method.startswith("Ensemble (C)"):
                    _score = strategy_C_ensemble(df_lot, _d)
                elif method.startswith("Ensemble (B + CatBoost)"):
                    _score = ensemble_scores_ac_cb(df_lot, _d, cb_model, cb_cols, **_eparams_y)
                elif method.startswith("Ensemble (B + LightGBM)"):
                    _score = ensemble_scores_ac_lgb(df_lot, _d, lgbm_model, lgbm_cols, **_eparams_y)
                elif method.startswith("ML (CatBoost"):
                    _score, _err2y = cb_scores_for_target(df_lot, _d, cb_model, cb_cols)
                else:
                    _score, _err2y = lgbm_scores_for_target(df_lot, _d, lgbm_model, lgbm_cols)

                if _score is None:
                    _yn = "N"
                else:
                    _picks = set(_score.head(k).index.tolist())
                    _yn = "Y" if _actual in _picks else "N"

                _yn_year_lines.append(_yn)

                _bg = "#d4edda" if _yn == "Y" else "#f8d7da"
                _fg = "#155724" if _yn == "Y" else "#721c24"
                _rows_year_html.append(
                    f"<tr><td class='dt'>{pd.to_datetime(_d).date()}</td>"
                    f"<td class='yn' style='background:{_bg};color:{_fg};text-align:center;font-weight:600;'>{_yn}</td></tr>"
                )

            _table_year_html = (
                "<table id='ynYearTable'><thead><tr><th class='dt'>Date</th><th>In Top-K</th></tr></thead><tbody>"
                + "".join(_rows_year_html) + "</tbody></table>"
            )

            table_payload_y = json.dumps(_table_year_html)
            plain_payload_y = json.dumps("\n".join(_yn_year_lines))

            html_year = """
<style>
#ynYearTable {
  border-collapse: collapse;
  font-family: monospace;
  width: 100%;
}
#ynYearTable th, #ynYearTable td {
  border: 1px solid #ddd;
  padding: 6px 10px;
}
#ynYearTable th.dt, #ynYearTable td.dt {
  background: #333333;
  color: #ffffff;
}
#ynYearScrollable {
  max-height: 560px;
  overflow: auto;
  border: 1px solid #eee;
  border-radius: 6px;
  padding: 6px;
  background: #fff;
}
</style>

<div style="margin:6px 0;">
  <div id="ynYearScrollable">__TABLE__</div>
  <div style="margin-top:8px;">
    <button id="copyYearHtmlBtn" style="padding:6px 10px; border-radius:6px; cursor:pointer;">Copy colored table</button>
    <button id="copyYearPlainBtn" style="padding:6px 10px; border-radius:6px; cursor:pointer; margin-left:6px;">Copy Y/N (plain)</button>
    <span id="copiedYearMsg" style="margin-left:8px;color:green;display:none;">Copied!</span>
  </div>
</div>

<script>
  const htmlTableYear = __TABLE_JSON__;
  const plainYear = __PLAIN_JSON__;

  document.getElementById('ynYearScrollable').innerHTML = htmlTableYear;

  async function copyHTMLWithFallback(html, plain) {
    if (navigator.clipboard && window.ClipboardItem) {
      try {
        const item = new ClipboardItem({
          'text/html': new Blob([html], {type: 'text/html'}),
          'text/plain': new Blob([plain], {type: 'text/plain'}),
        });
        await navigator.clipboard.write([item]);
        return true;
      } catch (e) {}
    }
    try {
      const hidden = document.createElement('div');
      hidden.contentEditable = 'true';
      hidden.style.position = 'fixed';
      hidden.style.left = '-9999px';
      hidden.innerHTML = html;
      document.body.appendChild(hidden);
      const range = document.createRange();
      range.selectNodeContents(hidden);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand('copy');
      document.body.removeChild(hidden);
      return true;
    } catch (e) {
      return false;
    }
  }

  document.getElementById('copyYearHtmlBtn').addEventListener('click', async () => {
    const ok = await copyHTMLWithFallback(htmlTableYear, plainYear);
    const note = document.getElementById('copiedYearMsg');
    note.textContent = ok ? 'Copied colored table!' : 'Copy failed';
    note.style.display = 'inline';
    setTimeout(() => note.style.display = 'none', 1500);
  });

  document.getElementById('copyYearPlainBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(plainYear);
    } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = plainYear;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    const note = document.getElementById('copiedYearMsg');
    note.textContent = 'Copied plain Y/N!';
    note.style.display = 'inline';
    setTimeout(() => note.style.display = 'none', 1500);
  });
</script>
""".replace("__TABLE__", _table_year_html).replace("__TABLE_JSON__", table_payload_y).replace("__PLAIN_JSON__", plain_payload_y)

            st_html(html_year, height=640)

            # Yearly totals (inline summary)
            _y_count = sum(1 for v in _yn_year_lines if str(v).strip().upper() == 'Y')
            _n_count = sum(1 for v in _yn_year_lines if str(v).strip().upper() == 'N')
            _total   = len(_yn_year_lines)
            _hitrate = (100.0 * _y_count / _total) if _total else 0.0
            st.markdown(
                f"**Yearly totals:** "
                f"Y = **{_y_count}**, N = **{_n_count}**, Hit-rate = **{_hitrate:.1f}%** ({_y_count}/{_total})"
            )

except Exception:
    pass
