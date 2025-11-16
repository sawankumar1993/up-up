
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Optional

try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Lottery Rig Detector Pro", page_icon="🕵️‍♂️", layout="wide")
st.title("🕵️‍♂️ Lottery Rig Detector — Pro")
st.caption("Fair reference, uniformity tests, stake-aware rig detection, FUTURE prediction, softmin–mixture fitting, and backtesting.")

# =============================
# Core helpers
# =============================

LOTTERY_COLS_DEFAULT = ["DR","SG","FB","GZ","GL"]

def coerce_int_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.astype("Int64")

def build_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
        if d.notna().any():
            df["Date"] = d.dt.date
    if "Date" not in df.columns or df["Date"].isna().all():
        if set(["Year","Month","Day"]).issubset(df.columns):
            d = pd.to_datetime(df[["Year","Month","Day"]].rename(columns={"Year":"year","Month":"month","Day":"day"}), errors="coerce")
            df["Date"] = d.dt.date
        else:
            df["Date"] = pd.NaT
    return df

def clean_outcomes_df(raw: pd.DataFrame, lotteries: Optional[List[str]] = None) -> pd.DataFrame:
    df = raw.copy()
    df = build_date_column(df)
    if lotteries is None:
        lotteries = [c for c in df.columns if c in LOTTERY_COLS_DEFAULT]
    for c in lotteries:
        if c in df.columns:
            df[c] = coerce_int_series(df[c])
            df.loc[~df[c].isna(), c] = df.loc[~df[c].isna(), c].clip(0,99)
    if "Date" in df.columns:
        df = df[df["Date"].notna()]
    if lotteries:
        df = df.dropna(subset=lotteries, how="all")
    return df

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    dti = pd.to_datetime(df["Date"])
    out = df.copy()
    out["Year"] = dti.dt.year
    out["Month"] = dti.dt.to_period("M").astype(str)
    iso = dti.dt.isocalendar()
    out["ISOYear"] = iso.year.astype(int)
    out["Week"] = iso.week.astype(int)
    out["Day"] = dti.dt.date
    out["DoW"] = dti.dt.day_name()
    return out

def date_range_df(start: date, end: date) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"Date": idx.date})

def generate_fair_draws(dates: pd.Series, lotteries: List[str], seed: Optional[int]=None, draws_per_day:int=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dt in dates:
        for _ in range(draws_per_day):
            row = {"Date": dt}
            for L in lotteries:
                row[L] = int(rng.integers(0,100))
            rows.append(row)
    return pd.DataFrame(rows)

def counts_0_99(series: pd.Series) -> np.ndarray:
    counts = np.zeros(100, dtype=int)
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    s = s[(s>=0)&(s<=99)]
    vc = s.value_counts()
    for num, cnt in vc.items():
        counts[int(num)] = int(cnt)
    return counts

def chisq_stat_and_pvalue(counts: np.ndarray, k:int=100) -> Tuple[float,float]:
    n = counts.sum()
    if n==0:
        return float("nan"), float("nan")
    expected = np.full(k, n/k)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = ((counts - expected)**2 / expected).sum()
    df = k-1
    if SCIPY_AVAILABLE and np.isfinite(chi2_stat):
        p_value = 1.0 - chi2.cdf(chi2_stat, df)
    else:
        if not np.isfinite(chi2_stat):
            return chi2_stat, float("nan")
        z = ((chi2_stat/df)**(1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
        try:
            from math import erfc, sqrt
            p_value = 0.5 * erfc(z / math.sqrt(2.0))
        except Exception:
            p_value = float("nan")
    return float(chi2_stat), float(p_value)

# =============================
# Stakes
# =============================

def parse_stake_csv(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None: return None
    try:
        raw = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0); raw = pd.read_csv(uploaded, encoding="latin-1")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = build_date_column(raw)

    if {"Date","Lottery","Number","Stake"}.issubset(set(raw.columns)):
        df = raw[["Date","Lottery","Number","Stake"]].copy()
        df = df.dropna(subset=["Date","Lottery","Number","Stake"])
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
        df["Stake"] = pd.to_numeric(df["Stake"], errors="coerce")
        df = df.dropna(subset=["Number","Stake"])
        df["Number"] = df["Number"].astype(int).clip(0,99)
        df["Stake"] = df["Stake"].astype(float).clip(lower=0)
        df["Lottery"] = df["Lottery"].astype(str).upper()
        return df

    num_cols = [c for c in raw.columns if c.isdigit() and 0<=int(c)<=99]
    if ("Date" in raw.columns) and ("Lottery" in raw.columns) and (len(num_cols)>=50):
        keep = ["Date","Lottery"] + sorted(num_cols, key=lambda x: int(x))
        dfw = raw[keep].copy()
        stake_long = dfw.melt(id_vars=["Date","Lottery"], var_name="Number", value_name="Stake")
        stake_long["Number"] = pd.to_numeric(stake_long["Number"], errors="coerce").astype(int)
        stake_long["Stake"] = pd.to_numeric(stake_long["Stake"], errors="coerce").fillna(0.0).astype(float).clip(lower=0)
        stake_long["Lottery"] = stake_long["Lottery"].astype(str).upper()
        stake_long = stake_long[(stake_long["Number"]>=0)&(stake_long["Number"]<=99)]
        stake_long = stake_long.dropna(subset=["Date"])
        return stake_long

    return None

def stakes_for_day_lottery(stake_long: pd.DataFrame, day: date, lottery: str) -> Optional[np.ndarray]:
    if stake_long is None: return None
    df = stake_long[(stake_long["Date"]==day) & (stake_long["Lottery"].str.upper()==str(lottery).upper())]
    if df.empty: return None
    arr = np.zeros(100, dtype=float)
    use_col = 'StakePrep' if 'StakePrep' in df.columns else 'Stake'
    for r in df.itertuples(index=False):
        arr[int(getattr(r, 'Number'))] += float(getattr(r, use_col))
    return arr

# =============================
# Proxy model
# =============================

def compute_proxy_stakes_for_day(day: date, lottery: str, history_df: pd.DataFrame,
                                 w_bday: float, w_mult5: float, w_lucky: float,
                                 w_hot: float, hot_window: int, w_avoid_last: float,
                                 lucky_list: List[int]) -> np.ndarray:
    S = np.ones(100, dtype=float)
    # Normalize history_df to long format with columns ['Date','Lottery','Win']
    hist_df = history_df.copy() if history_df is not None else pd.DataFrame(columns=['Date','Lottery','Win'])
    if 'Lottery' not in hist_df.columns or 'Win' not in hist_df.columns:
        if 'Date' in hist_df.columns:
            value_cols = [c for c in hist_df.columns if c != 'Date']
            hist_df = hist_df.melt(id_vars=['Date'], value_vars=value_cols, var_name='Lottery', value_name='Win')
        else:
            hist_df = pd.DataFrame(columns=['Date','Lottery','Win'])
    if 'Date' in hist_df.columns:
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors='coerce').dt.date
    if 'Win' in hist_df.columns:
        hist_df['Win'] = pd.to_numeric(hist_df['Win'], errors='coerce')
    if w_bday: S[1:32] += w_bday
    if w_mult5: S[::5] += w_mult5
    if w_lucky and lucky_list:
        for n in lucky_list:
            if 0<=n<=99: S[n] += w_lucky
    if w_hot and hot_window>0:
        cutoff = day - timedelta(days=hot_window)
        hist = hist_df[(hist_df["Lottery"].str.upper()==str(lottery).upper()) & (hist_df["Date"]<day) & (hist_df["Date"]>=cutoff)]
        if not hist.empty:
            counts = np.zeros(100, dtype=int)
            vs = pd.to_numeric(hist["Win"], errors="coerce").dropna().astype(int)
            vs = vs[(vs>=0)&(vs<=99)]
            vc = vs.value_counts()
            for num, cnt in vc.items():
                counts[int(num)] = int(cnt)
            S += w_hot * counts
    if w_avoid_last:
        prev = hist_df[(hist_df["Lottery"].str.upper()==str(lottery).upper()) & (hist_df["Date"]<day)].sort_values("Date").tail(1)
        if len(prev):
            last = prev["Win"].iloc[0]
            if pd.notna(last):
                try:
                    last = int(last)
                    if 0<=last<=99: S[last] = max(1e-6, S[last]-w_avoid_last)
                except Exception: pass
    return np.clip(S, 1e-6, None)

def bottom_k_from_stakes(S: np.ndarray, k:int=20) -> Tuple[List[int], float]:
    if S is None: return [], float("nan")
    idx = np.argsort(S)
    if k<=0: return [], 0.0
    if len(S)<=k:
        chosen = list(idx.tolist())
        return chosen, len(chosen)/len(S)
    thresh = S[idx[k-1]]
    chosen = [int(i) for i in idx if S[int(i)] <= thresh + 1e-12]
    return chosen, len(chosen)/len(S)

def min_set_from_stakes(S: np.ndarray) -> Tuple[List[int], float]:
    if S is None: return [], float("nan")
    m = float(S.min())
    chosen = [int(i) for i,v in enumerate(S) if abs(v-m)<=1e-12]
    return chosen, len(chosen)/len(S)

# =============================
# Softmin–mixture
# =============================

def softmin_probs(S: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0: tau = 1e-6
    x = -np.array(S, dtype=float) / float(tau)
    x = x - x.max()
    w = np.exp(x)
    p = w / w.sum()
    return p

def loglik_softmix(win: int, S: np.ndarray, alpha: float, tau: float) -> float:
    p_soft = softmin_probs(S, tau)
    p = alpha * p_soft + (1.0 - alpha) * (1.0/100.0)
    p_win = p[int(win)] if 0<=int(win)<=99 else 1e-12
    return float(np.log(max(p_win, 1e-12)))

def fit_softmix_alpha_tau(history_rows: List[Dict], alpha_grid=None, tau_grid=None) -> Tuple[float,float,float]:
    if alpha_grid is None: alpha_grid = np.linspace(0.0, 1.0, 21)
    if tau_grid is None: tau_grid = np.geomspace(0.05, 5.0, 25)
    best = (-1e18, 0.0, 1.0)
    for a in alpha_grid:
        for t in tau_grid:
            ll = 0.0
            for row in history_rows:
                S = row["S"]; win = row["win"]
                if S is None or np.any(~np.isfinite(S)) or win is None: continue
                ll += loglik_softmix(int(win), S, a, t)
            if ll > best[0]:
                best = (ll, a, t)
    return best[1], best[2], best[0]


# =============================
# Reliability helpers (added)
# =============================
def preprocess_stakes(stake_long: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'StakePrep' per (Date, Lottery) using share-of-wallet + log smoothing.
    """
    if stake_long is None or len(stake_long)==0:
        return stake_long
    df = stake_long.copy()
    # Cap outliers within each group using 95th percentile
    def _cap(group):
        x = group["Stake"].astype(float)
        if len(x) >= 5:
            hi = x.quantile(0.95)
            x = x.clip(upper=float(hi))
        return x
    df["StakeCapped"] = df.groupby(["Date","Lottery"], group_keys=False).apply(_cap)
    # Share-of-wallet normalization
    sums = df.groupby(["Date","Lottery"])["StakeCapped"].transform("sum").replace(0, 1.0)
    df["StakeShare"] = (df["StakeCapped"] / sums).astype(float)
    # Log smoothing
    df["StakeLog"] = np.log1p(df["StakeCapped"]).astype(float)
    def _norm01(g):
        x = g["StakeLog"].values.astype(float)
        lo, hi = float(np.min(x)), float(np.max(x))
        g["StakeLog01"] = 0.0 if hi-lo < 1e-12 else (x - lo) / (hi - lo)
        return g
    df = df.groupby(["Date","Lottery"], group_keys=False).apply(_norm01)
    df["StakePrep"] = 0.7*df["StakeShare"] + 0.3*df["StakeLog01"]
    return df

def choose_set_by_target_mean(incl_prob: np.ndarray, target: float, k_min:int=12, k_max:int=60) -> list:
    order = np.argsort(-incl_prob)
    cumsum = np.cumsum(incl_prob[order])
    ks = np.arange(1, len(order)+1)
    means = cumsum / ks
    k_lo = max(1, int(k_min))
    k_hi = min(len(order), int(k_max))
    idx = None
    for k in range(k_lo, k_hi+1):
        if means[k-1] >= float(target):
            idx = k; break
    if idx is None:
        idx = k_hi
    return sorted([int(i) for i in order[:idx]])

def blend_with_proxy(S_real: Optional[np.ndarray], S_proxy: np.ndarray, min_cov: float=0.3, w_proxy: float=0.3) -> np.ndarray:
    if S_real is None:
        S_real = np.zeros(100, dtype=float); cov = 0.0
    else:
        cov = float(np.count_nonzero(np.array(S_real)>0)) / 100.0
    w = float(w_proxy) if cov < float(min_cov) else 0.0
    S = (1.0 - w) * np.array(S_real, dtype=float) + w * np.array(S_proxy, dtype=float)
    return np.clip(S, 1e-6, None)

# =============================
# RSMO-25 (Robust Softmix Optimizer — 25) helpers
# =============================
def softmix_probs_from_stakes(S: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """Estimated per-number win probabilities under a soft-min mixture model."""
    S = np.array(S, dtype=float)
    S = np.clip(S, 1e-9, None)
    inv = (1.0 / S) ** float(alpha)
    inv_sum = inv.sum()
    p = (inv / inv_sum) if inv_sum > 0 and np.isfinite(inv_sum) else (np.ones_like(inv) / len(inv))
    # tau = weight on model, (1 - tau) on uniform
    p = float(tau) * p + (1.0 - float(tau)) * (np.ones_like(p) / len(p))
    p = np.clip(p, 1e-15, None)
    return p / p.sum()

def choose_top25_target80(
    S: np.ndarray,
    alpha_default: float,
    tau_default: float,
    noise_sigma: float,
    target: float = 0.80,
    seed: int = 12345,
) -> tuple[list[int], float, int]:
    """
    Return (chosen_25, expected_hit_prob, pool_size_used).
    Strategy:
      - Candidate pool sizes: 40 -> 60 -> 80 -> 100 (least-staked)
      - Robust P(win) = average softmix over {0.9*α, α, 1.1*α} AND 3 noise draws per α
      - Pick the top-25 by robust P(win) inside the pool
      - If expected hit < target, widen pool (still 25 elements)
    """
    S = np.array(S, dtype=float)
    S = np.clip(S, 1e-9, None)
    order_by_stake = np.argsort(S)  # least-staked first
    alphas = [max(0.05, alpha_default*0.9), alpha_default, min(2.0, alpha_default*1.1)]
    rng = np.random.default_rng(seed)

    def robust_probs(baseS: np.ndarray) -> np.ndarray:
        ps = []
        for a in alphas:
            # deterministic
            ps.append(softmix_probs_from_stakes(baseS, a, tau_default))
            # noisy variants
            for _ in range(3):
                noise = rng.lognormal(mean=0.0, sigma=float(noise_sigma), size=100)
                ps.append(softmix_probs_from_stakes(baseS * noise, a, tau_default))
        P = np.mean(np.vstack(ps), axis=0)
        P = np.clip(P, 1e-15, None)
        return P / P.sum()

    P_full = robust_probs(S)

    chosen = None
    expected_hit = 0.0
    used_pool = 40
    for M in [40, 60, 80, 100]:
        pool = order_by_stake[:M]
        # rank by robust P(win); tie-break: lower stake, then index
        cand_sorted = sorted(pool, key=lambda i: (-P_full[i], S[i], i))
        chosen = sorted(int(i) for i in cand_sorted[:25])
        expected_hit = float(P_full[chosen].sum())
        used_pool = M
        if expected_hit >= float(target) or M == 100:
            break
    return chosen, expected_hit, used_pool
# =============================
# UI — Sidebar
# =============================

st.sidebar.header("1) Upload outcomes CSV")
uploaded_outcomes = st.sidebar.file_uploader("Outcomes CSV", type=["csv"], key="outcomes")

st.sidebar.header("2) Optional: Upload stakes CSV")
uploaded_stakes = st.sidebar.file_uploader("Stakes CSV (long or wide)", type=["csv"], key="stakes")

st.sidebar.header("3) Choose lotteries")
lotteries = st.sidebar.multiselect("Lottery columns", LOTTERY_COLS_DEFAULT, default=LOTTERY_COLS_DEFAULT)

st.sidebar.header("4) Fair reference generation")
seed = st.sidebar.number_input("Random seed (optional)", 0, 10**9, 42, 1)
c1,c2 = st.sidebar.columns(2)
with c1:
    start_date = st.sidebar.date_input("Start date", value=date(2025,7,1))
with c2:
    end_date = st.sidebar.date_input("End date", value=date(2025,7,31))
draws_per_day = st.sidebar.number_input("Draws per day (per lottery)", 1, 50, 1, 1)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

st.sidebar.header("5) Proxy popularity model (if no stakes)")
use_proxy = st.sidebar.checkbox("Use proxy model when stakes missing", True)
w_bday = st.sidebar.slider("Weight: Birthday (1..31)", 0.0, 2.0, 0.6, 0.1)
w_mult5 = st.sidebar.slider("Weight: Multiples of 5", 0.0, 2.0, 0.4, 0.1)
w_lucky = st.sidebar.slider("Weight: Lucky numbers", 0.0, 2.0, 0.5, 0.1)
lucky_str = st.sidebar.text_input("Lucky numbers", "7,9,11,13")
lucky_list = []
for tok in lucky_str.split(","):
    tok = tok.strip()
    if tok:
        try:
            v = int(tok); 
            if 0<=v<=99: lucky_list.append(v)
        except: pass
w_hot = st.sidebar.slider("Weight: Hotness (last k days)", 0.0, 1.0, 0.2, 0.05)
hot_window = st.sidebar.number_input("k days for hotness", 1, 60, 14, 1)
w_avoid_last = st.sidebar.slider("Weight: Avoid last winner", 0.0, 2.0, 0.7, 0.1)

st.sidebar.header("6) Advanced — Softmin–mixture fitting")
fit_window_days = st.sidebar.slider("Fitting window (days)", 7, 120, 30, 1)
alpha_default = st.sidebar.slider("Default α (if not enough data)", 0.0, 1.0, 0.3, 0.05)
tau_default = st.sidebar.slider("Default τ (if not enough data)", 0.01, 5.0, 0.6, 0.01)

st.sidebar.header("7) Backtest")
enable_backtest = st.sidebar.checkbox("Enable Backtest tab", value=True)

# =============================
# Data prep
# =============================

fair_dates_df = date_range_df(start_date, end_date)
fair_df = generate_fair_draws(fair_dates_df["Date"], lotteries, seed=seed, draws_per_day=draws_per_day)
fair_df = add_time_columns(fair_df)

st.subheader("Fair reference data (generated)")
st.dataframe(fair_df.head(20), use_container_width=True)

out_df = None
if uploaded_outcomes is not None:
    try:
        out_df = clean_outcomes_df(pd.read_csv(uploaded_outcomes), lotteries=lotteries)
    except UnicodeDecodeError:
        uploaded_outcomes.seek(0); out_df = clean_outcomes_df(pd.read_csv(uploaded_outcomes, encoding="latin-1"), lotteries=lotteries)
    out_df = out_df[["Date"] + [c for c in lotteries if c in out_df.columns]]
    out_df = add_time_columns(out_df)

st.subheader("Your uploaded outcomes")
if out_df is not None and len(out_df):
    st.success(f"Loaded {len(out_df)} rows."); st.dataframe(out_df.head(20), use_container_width=True)
else:
    st.info("No outcomes uploaded. You can still explore with fair reference.")

stake_long = parse_stake_csv(uploaded_stakes)
stake_long = preprocess_stakes(stake_long)
st.subheader("Your uploaded stakes")
if stake_long is not None and len(stake_long):
    st.success(f"Loaded {len(stake_long)} stake rows."); st.dataframe(stake_long.head(20), use_container_width=True)
else:
    if uploaded_stakes is not None:
        st.warning("Could not parse stakes CSV. Use long (Date,Lottery,Number,Stake) or wide (Date,Lottery,0..99).")
    else:
        st.info("No stakes CSV uploaded. Proxy model will be used if enabled.")

st.divider()

# =============================
# Slice selection & uniformity
# =============================

st.subheader("Select a time slice to analyze")
df_for_slice = out_df if (out_df is not None and len(out_df)) else fair_df.copy()
if len(df_for_slice)==0: st.warning("No data to slice."); st.stop()
df_for_slice = add_time_columns(df_for_slice)

granularity = st.selectbox("Analyze by", ["Day","Week (ISO)","Month","Year","Day of Week"])

if granularity == "Day":
    options = sorted(df_for_slice["Day"].unique().tolist())
    pick = st.selectbox("Pick a day", options, index=0)
elif granularity == "Week (ISO)":
    tmp = df_for_slice[["ISOYear","Week"]].drop_duplicates().sort_values(["ISOYear","Week"])
    options = [(int(r.ISOYear), int(r.Week)) for r in tmp.itertuples(index=False)]
    pick = st.selectbox("Pick ISO week (year, week)", options, index=0, format_func=lambda t: f"{t[0]}-W{t[1]:02d}")
elif granularity == "Month":
    options = sorted(df_for_slice["Month"].unique().tolist())
    pick = st.selectbox("Pick a month", options, index=0)
elif granularity == "Year":
    options = sorted(df_for_slice["Year"].unique().tolist())
    pick = st.selectbox("Pick a year", options, index=0)
else:
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    options = [d for d in dow_order if d in set(df_for_slice["DoW"].unique())]
    pick = st.selectbox("Pick a day of week", options, index=0)

st.caption(f"Analyzing slice: **{pick}** (by **{granularity}**)")
slice_df = df_for_slice.copy()
if granularity == "Day":
    slice_df = slice_df[slice_df["Day"] == pick]
elif granularity == "Week (ISO)":
    iso_year, iso_week = pick
    slice_df = slice_df[(slice_df["ISOYear"]==iso_year) & (slice_df["Week"]==iso_week)]
elif granularity == "Month":
    slice_df = slice_df[slice_df["Month"]==pick]
elif granularity == "Year":
    slice_df = slice_df[slice_df["Year"]==pick]
else:
    slice_df = slice_df[slice_df["DoW"]==pick]

st.write(f"{len(slice_df)} rows in selected slice.")
if len(slice_df)==0: st.warning("No rows in the selected slice."); st.stop()

st.subheader("Uniformity (0–99) — chi-square vs fair")
def analyze_uniformity(slice_df: pd.DataFrame, lotteries: List[str]) -> pd.DataFrame:
    rows = []
    for L in lotteries:
        if L not in slice_df.columns: continue
        counts = counts_0_99(slice_df[L]); n = counts.sum()
        chi2_stat, p_value = chisq_stat_and_pvalue(counts, 100)
        rows.append({"Lottery":L, "Draws in slice":int(n), "Chi-square":round(chi2_stat,3) if chi2_stat==chi2_stat else np.nan,
                     "p-value":p_value, "Anomaly score (1 - p)": (1.0-p_value) if p_value==p_value else np.nan})
    return pd.DataFrame(rows)
uniform_df = analyze_uniformity(slice_df, lotteries)
st.dataframe(uniform_df, use_container_width=True)

# =============================
# History for proxy hotness / avoid-last
# =============================

hist_rows = []
if out_df is not None and len(out_df):
    for r in out_df.itertuples(index=False):
        for L in lotteries:
            if L in out_df.columns and not pd.isna(getattr(r, L)):
                hist_rows.append({"Date": r.Date, "Lottery": L, "Win": getattr(r, L)})
history = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame(columns=["Date","Lottery","Win"])

# =============================
# Fit softmin–mixture (α, τ)
# =============================

st.divider()
st.header("⚙️ Softmin–mixture fitting (α, τ)")

fit_rows = []
if out_df is not None and len(out_df):
    end_cut = slice_df["Day"].max()
    start_cut = end_cut - timedelta(days=int(fit_window_days))
    for r in out_df[(out_df["Day"]<end_cut) & (out_df["Day"]>=start_cut)].itertuples(index=False):
        dt = r.Date
        for L in lotteries:
            if L in out_df.columns and not pd.isna(getattr(r, L)):
                S = stakes_for_day_lottery(stake_long, dt, L)
                if S is None and use_proxy:
                    S = compute_proxy_stakes_for_day(dt, L, history, w_bday, w_mult5, w_lucky, w_hot, hot_window, w_avoid_last, lucky_list)
                if S is None: continue
                fit_rows.append({"S":S, "win": int(getattr(r, L)), "Date":dt, "Lottery":L})

if len(fit_rows) < 10:
    st.info("Not enough historical rows to fit; using defaults.")
    alpha_hat, tau_hat = alpha_default, tau_default
    loglik = float("nan")
else:
    alpha_hat, tau_hat, loglik = fit_softmix_alpha_tau(fit_rows)
st.write(f"**α̂** (rig rate): {alpha_hat:.2f}   |   **τ̂** (softness): {tau_hat:.3f}   |   log-likelihood: {loglik if loglik==loglik else '—'}")

# =============================
# 🔮 Future prediction
# =============================

st.divider()
st.header("🔮 Future prediction — Least-staked numbers")

c1,c2,c3 = st.columns(3)
with c1:
    future_date = st.date_input("Future date", value=max(date.today(), end_date + timedelta(days=1)))
with c2:
    future_lottery = st.selectbox("Lottery", lotteries if lotteries else LOTTERY_COLS_DEFAULT, index=0)
with c3:
    sims = st.number_input("Simulations for inclusion probabilities", 200, 20000, 3000, 100)

noise_sigma = st.slider("Uncertainty (log-normal sigma)", 0.0, 1.0, 0.25, 0.05)

strategy = st.selectbox("Set Size Strategy", ["Tight (~24)","Balanced (~32)","Wide (~40)","By inclusion-prob threshold","Target hit-rate (~80%)"])
incl_thresh = 0.25
if strategy=="By inclusion-prob threshold":
    incl_thresh = st.slider("Inclusion probability ≥", 0.05, 0.95, 0.35, 0.05)

# Added: target hit-rate control and leakage-safe toggle
target_mean = 0.80
leakage_safe = st.checkbox("Leakage-safe mode (use lagged stakes)", value=True)
if strategy=="Target hit-rate (~80%)":
    target_mean = st.slider("Target mean inclusion probability", 0.50, 0.95, 0.80, 0.01)
S_real = stakes_for_day_lottery(stake_long, (future_date - timedelta(days=1)) if leakage_safe else future_date, future_lottery)
S_proxy = compute_proxy_stakes_for_day(future_date, future_lottery, out_df if out_df is not None else fair_df,
                                       w_bday, w_mult5, w_lucky, w_hot, hot_window, w_avoid_last, lucky_list)
S_future = blend_with_proxy(S_real, S_proxy, min_cov=0.3, w_proxy=0.3)
b20_set, p_base = bottom_k_from_stakes(S_future, 20)
st.markdown(f"**Baseline bottom-20 (ties):** size={len(b20_set)} → baseline={len(b20_set)/100:.2%}")
st.code(", ".join(str(x) for x in sorted(b20_set)) if b20_set else "—")

rng = np.random.default_rng(12345)
incl = np.zeros(100, dtype=int)
base = np.array(S_future, dtype=float); base = np.clip(base, 1e-6, None)
for _ in range(int(sims)):
    noise = rng.lognormal(mean=0.0, sigma=float(noise_sigma), size=100)
    S_pert = base * noise
    bset, _ = bottom_k_from_stakes(S_pert, 20)
    incl[bset] += 1
incl_prob = incl / sims

def choose_by_strategy(incl_prob: np.ndarray, strategy: str, default_sizes=(24,32,40), thresh=0.35):
    order = np.argsort(-incl_prob)
    if strategy=="Tight (~24)":
        N = default_sizes[0]
        return sorted(list(order[:N]))
    elif strategy=="Balanced (~32)":
        N = default_sizes[1]
        return sorted(list(order[:N]))
    elif strategy=="Wide (~40)":
        N = default_sizes[2]
        return sorted(list(order[:N]))
    elif strategy=="Target hit-rate (~80%)":
        # Uses global 'target_mean' configured via the UI
        try:
            return choose_set_by_target_mean(incl_prob, float(target_mean))
        except Exception:
            # Fallback to balanced if something goes wrong
            N = default_sizes[1]
            return sorted(list(order[:N]))
    else:
        # "By inclusion-prob threshold"
        chosen = [int(i) for i in range(100) if incl_prob[i] >= thresh]
        if not chosen:
            return list(order[:24])
        return sorted(chosen, key=lambda i: (-incl_prob[i], i))
pred_set = choose_by_strategy(incl_prob, strategy, thresh=incl_thresh)
st.subheader(
"Predicted least-staked set (strategy-adjusted)")
st.markdown(f"**Set size:** {len(pred_set)} • **Mean inclusion prob:** {np.mean(incl_prob[pred_set]):.2%}")
st.code(", ".join(str(x) for x in sorted(pred_set)))


# === RSMO-25: fixed top-25 optimized for ~80% expected hit ===
with st.expander("Target-80 (Top-25 least-staked) — robust optimizer", expanded=False):
    try:
        t80_set, t80_hit, t80_pool = choose_top25_target80(
            S_future,
            float(alpha_default),
            float(tau_default),
            float(noise_sigma),
            target=0.80,
        )
        st.markdown(
            f"**Set size:** 25  •  **Candidate pool used:** bottom {t80_pool} by stake  •  "
            f"**Expected hit (model):** {t80_hit:.2%}"
        )
        st.code(", ".join(str(x) for x in t80_set))
        try:
            overlap = set(t80_set) & set(pred_set)
            st.markdown(f"Overlap with current strategy set: {len(overlap)}")
            if overlap:
                st.code(", ".join(str(x) for x in sorted(overlap)))
        except Exception:
            pass
    except Exception as e:
        st.error(f"RSMO-25 failed: {e}")
pred_df = pd.DataFrame({
    "number": np.arange(100, dtype=int),
    "stake_score": base,
    "incl_prob": incl_prob,
    "in_pred_set": [int(n in set(pred_set)) for n in range(100)]
}).sort_values(["in_pred_set","incl_prob","number"], ascending=[False, False, True])
st.dataframe(pred_df.head(40), use_container_width=True)

def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
st.download_button("Download prediction CSV", data=to_csv(pred_df), file_name="future_prediction.csv", mime="text/csv")

# =============================
# 📈 Backtest (walk-forward)
# =============================

if enable_backtest and out_df is not None and len(out_df):
    st.divider()
    st.header("📈 Backtest — walk-forward")
    c1,c2 = st.columns(2)
    with c1:
        bt_start = st.date_input("Backtest start", value=(out_df["Day"].min() if "Day" in out_df.columns else date(2025,7,1)))
    with c2:
        bt_end = st.date_input("Backtest end", value=(out_df["Day"].max() if "Day" in out_df.columns else date(2025,7,31)))

    bt_lottery = st.selectbox("Lottery to backtest", lotteries if lotteries else LOTTERY_COLS_DEFAULT, index=0)

    if bt_start >= bt_end:
        st.warning("Backtest start must be before end.")
    else:
        days = pd.date_range(bt_start, bt_end, freq="D").date
        hist_bt = []
        for dt in days:
            # Fit params on prior window
            win_rows = []
            start_fit = dt - timedelta(days=int(fit_window_days))
            df_hist = out_df[(out_df["Day"]<dt) & (out_df["Day"]>=start_fit)]
            for r in df_hist.itertuples(index=False):
                if bt_lottery in out_df.columns and not pd.isna(getattr(r, bt_lottery)):
                    S = stakes_for_day_lottery(stake_long, r.Date, bt_lottery)
                    if S is None and use_proxy:
                        S = compute_proxy_stakes_for_day(r.Date, bt_lottery, history, w_bday, w_mult5, w_lucky, w_hot, hot_window, w_avoid_last, lucky_list)
                    if S is None: continue
                    win_rows.append({"S":S, "win": int(getattr(r, bt_lottery))})
            if len(win_rows) >= 10:
                a_hat, t_hat, _ = fit_softmix_alpha_tau(win_rows)
            else:
                a_hat, t_hat = alpha_default, tau_default

            # Predict set for dt
            S_real_bt = stakes_for_day_lottery(stake_long, (dt - timedelta(days=1)) if leakage_safe else dt, bt_lottery)
            S_proxy_bt = compute_proxy_stakes_for_day(dt, bt_lottery, out_df if out_df is not None else fair_df,
                                         w_bday, w_mult5, w_lucky, w_hot, hot_window, w_avoid_last, lucky_list)
            S_future_bt = blend_with_proxy(S_real_bt, S_proxy_bt, min_cov=0.3, w_proxy=0.3)
            rng = np.random.default_rng(12345)
            incl_bt = np.zeros(100, dtype=int)
            base_bt = np.array(S_future_bt, dtype=float); base_bt = np.clip(base_bt, 1e-6, None)
            for _ in range(1000):
                noise = rng.lognormal(mean=0.0, sigma=0.25, size=100)
                S_pert = base_bt * noise
                bset_bt, _ = bottom_k_from_stakes(S_pert, 20)
                incl_bt[bset_bt] += 1
            incl_prob_bt = incl_bt / 1000.0
            def choose_by_strategy_local(incl_prob):
                return set(choose_by_strategy(incl_prob, strategy, thresh=incl_thresh))
            pred_set_bt = choose_by_strategy_local(incl_prob_bt)

            actual_row = out_df[out_df["Day"]==dt]
            if len(actual_row):
                win = actual_row[bt_lottery].dropna()
                if len(win):
                    w = int(win.iloc[0])
                    hit = int(w in pred_set_bt)
                    hist_bt.append({"Date":dt, "alpha_hat":a_hat, "tau_hat":t_hat, "set_size":len(pred_set_bt), "hit":hit})

        if hist_bt:
            hdf = pd.DataFrame(hist_bt)
            hit_rate = hdf["hit"].mean() if len(hdf) else float("nan")
            avg_size = hdf["set_size"].mean() if len(hdf) else float("nan")
            st.write(f"**Backtest results** — Hit-rate: {hit_rate:.2%}  |  Avg set size: {avg_size:.1f}")
            st.dataframe(hdf.tail(30), use_container_width=True)
        else:
            st.info("No backtest history produced (insufficient data or missing outcomes).")

# =============================
# Visuals
# =============================

st.subheader("Distributions vs uniform (0–99) — outcomes")
import altair as alt
def plot_dist(series: pd.Series, title: str):
    counts = counts_0_99(series)
    dfp = pd.DataFrame({"number": np.arange(100), "count": counts})
    expected = counts.sum()/100 if counts.sum()>0 else 0
    base = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("number:O", title="Number (0–99)"),
        y=alt.Y("count:Q", title="Observed count"),
        tooltip=["number","count"]
    ).properties(title=title, height=240)
    line = alt.Chart(pd.DataFrame({"y":[expected]})).mark_rule().encode(y="y:Q")
    return base + line

cols = st.columns(len(lotteries) if lotteries else 1)
for i, L in enumerate(lotteries):
    if L in slice_df.columns:
        with cols[i % len(cols)]:
            st.altair_chart(plot_dist(slice_df[L], f"{L}: observed vs expected mean"), use_container_width=True)
