######### PRE-PROCESSING

# Basic packets
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ARCH library
from arch import arch_model

#ARIMA library
from statsmodels.tsa.arima.model import ARIMA

#Loading the data
equity_pricing = pd.read_csv('Data/Equity_Pricing.csv')
equity_vol = pd.read_csv('Data/Equity_Vol.csv')
iv = pd.read_csv('Data/IV.csv')
option_contracts = pd.read_csv('Data/Option_Contracts.csv')
option_prices = pd.read_csv('Data/Option_Prices.csv')

# Stock split adjustment for NVDA (10-for-1 split)
# equity_pricing: adjust price columns to match yahoo_hist (already split-adjusted via auto_adjust=True)
cols_to_scale = ["open_", "high", "low", "close_", "bid", "ask"]
mask_nvda = equity_pricing["ticker"] == "NVDA"
equity_pricing.loc[mask_nvda, cols_to_scale] = equity_pricing.loc[mask_nvda, cols_to_scale] / 10

# option_contracts: adjust NVDA strikes by the same factor so ATM selection
# compares strikes and spot on the same scale
mask_nvda_oc = option_contracts["opraticker"] == "NVDA"
option_contracts.loc[mask_nvda_oc, "strike"] = option_contracts.loc[mask_nvda_oc, "strike"] / 10

tickers = ['AMZN', 'AAPL', 'C', 'XOM', 'JPM', 'LLY', 'NVDA', 'UNH', 'V', 'TSLA']

dec_start = "2023-12-01"
dec_end   = "2023-12-31"

hist_start = "2018-01-01"
hist_end   = "2024-01-31"

def yf_download_long(tickers, start, end):
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=(pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True
    )

    long = raw.stack(level=1).reset_index()

    if "Ticker" in long.columns:
        ticker_col = "Ticker"
    elif "level_1" in long.columns:
        ticker_col = "level_1"
    else:
        raise KeyError(f"Ticker column not found. Columns: {list(long.columns)}")

    if "Date" in long.columns:
        date_col = "Date"
    elif "level_0" in long.columns:
        date_col = "level_0"
    else:
        raise KeyError(f"Date column not found. Columns: {list(long.columns)}")

    long = long.rename(columns={
        ticker_col: "ticker",
        date_col: "date_",
        "Open": "open_",
        "High": "high",
        "Low": "low",
        "Close": "close_",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    # Remove 'adj_close' from the 'keep' list as auto_adjust=True makes 'close_' adjusted.
    keep = ["ticker", "date_", "open_", "high", "low", "close_", "volume"]
    missing = [c for c in keep if c not in long.columns]
    if missing:
        raise KeyError(f"Missing expected columns from yfinance output: {missing}. Have: {list(long.columns)}")

    return long[keep].sort_values(["ticker", "date_"]).reset_index(drop=True)

# Yahoo: Dec 2023 for matching + longer history for modeling
yahoo_dec  = yf_download_long(tickers, dec_start, dec_end)
yahoo_hist = yf_download_long(tickers, hist_start, hist_end)

# 1) Base df: keep your existing identifiers + dates (Dec 2023) from equity_vol
df = (
    equity_vol
    .groupby(["eqId", "date_"], as_index=False)
    .agg(
        ticker=("ticker", "first"),
        issuer=("issuer", "first"),
        issue=("issue", "first"),
        cusip=("cusip", "first"),
    )
    .sort_values(["eqId", "date_"])
)

df["date_"] = pd.to_datetime(df["date_"], errors="coerce").dt.normalize()

# 2) Build forward realized variance/vol targets from Yahoo close prices
y = yahoo_hist.copy()
y["date_"] = pd.to_datetime(y["date_"], errors="coerce").dt.normalize()
y = y.dropna(subset=["ticker", "date_", "close_"]).sort_values(["ticker", "date_"]).reset_index(drop=True)

g = y.groupby("ticker", group_keys=False)

# forward 1-day log return: r_{t+1} = log(C_{t+1}/C_t)
y["ret_fwd_1"] = np.log(g["close_"].shift(-1) / y["close_"])
y["sq_ret_fwd_1"] = y["ret_fwd_1"] ** 2

# future realized variance over next h days: sum_{i=1..h} r_{t+i}^2
y["rv_fwd_3"]  = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(3).sum())
y["rv_fwd_5"]  = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(5).sum())
y["rv_fwd_10"] = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(10).sum())

# convert to realized volatility (sqrt of variance)
y["vol_fwd_3"]  = np.sqrt(y["rv_fwd_3"])
y["vol_fwd_5"]  = np.sqrt(y["rv_fwd_5"])
y["vol_fwd_10"] = np.sqrt(y["rv_fwd_10"])

targets = y[["ticker", "date_", "vol_fwd_3", "vol_fwd_5", "vol_fwd_10"]].copy()

# 3) Merge targets into df (same df name)
df = df.merge(targets, on=["ticker", "date_"], how="left")

# (optional) check if any targets missing for your Dec dates
df[["vol_fwd_3","vol_fwd_5","vol_fwd_10"]].isna().sum()

# --- normalize dates for a clean merge ---
df["date_"] = pd.to_datetime(df["date_"], errors="coerce").dt.normalize()

y = yahoo_hist.copy()
y["date_"] = pd.to_datetime(y["date_"], errors="coerce").dt.normalize()
y = y.dropna(subset=["ticker", "date_"]).sort_values(["ticker", "date_"]).reset_index(drop=True)

g = y.groupby("ticker")

# --- Returns (log) ---
y["ret_1"]   = np.log(y["close_"] / g["close_"].shift(1))
y["ret_2"]   = np.log(g["close_"].shift(1) / g["close_"].shift(3))
y["ret_5"]   = np.log(g["close_"].shift(1) / g["close_"].shift(6))
y["ret_10"]  = np.log(g["close_"].shift(1) / g["close_"].shift(11))
y["ret_22"]  = np.log(g["close_"].shift(1) / g["close_"].shift(23))
y["ret_60"]  = np.log(g["close_"].shift(1) / g["close_"].shift(61))
y["ret_90"]  = np.log(g["close_"].shift(1) / g["close_"].shift(91))
y["ret_120"] = np.log(g["close_"].shift(1) / g["close_"].shift(121))
y["ret_150"] = np.log(g["close_"].shift(1) / g["close_"].shift(151))
y["ret_180"] = np.log(g["close_"].shift(1) / g["close_"].shift(181))
y["ret_200"] = np.log(g["close_"].shift(1) / g["close_"].shift(201))

# --- Range / intraday movement ---
y["hl_log"] = np.log(y["high"] / y["low"])
y["oc_log"] = np.log(y["close_"] / y["open_"])
y["co_log"] = np.log(y["open_"] / g["close_"].shift(1))

# --- Volume ---
vol = y["volume"].astype(float)
y["log_vol"]  = np.log(vol.where(vol > 0))
y["dlog_vol"] = g["log_vol"].diff()

# --- Rolling vol proxies (close-to-close), using only info up to t-1 ---
y["rv1"]  = y["ret_1"] ** 2
y["rv5"]  = g["rv1"].transform(lambda s: s.shift(1).rolling(5).sum())
y["rv22"] = g["rv1"].transform(lambda s: s.shift(1).rolling(22).sum())
y["vol5"]  = np.sqrt(y["rv5"])
y["vol22"] = np.sqrt(y["rv22"])

# --- Range-based rolling proxies ---
y["hl_5"]  = g["hl_log"].transform(lambda s: s.shift(1).rolling(5).mean())
y["hl_22"] = g["hl_log"].transform(lambda s: s.shift(1).rolling(22).mean())

# --- Momentum / trend (optional) ---
y["mom_5"]   = y["ret_5"]
y["mom_22"]  = y["ret_22"]
y["mom_200"] = y["ret_200"]

feat_cols = [
    "ticker","date_",
    "ret_1","ret_2","ret_5","ret_10","ret_22","ret_60","ret_90","ret_120","ret_150","ret_180","ret_200",
    "hl_log","oc_log","co_log",
    "log_vol","dlog_vol",
    "rv1","rv5","rv22","vol5","vol22",
    "hl_5","hl_22",
    "mom_5","mom_22","mom_200"
]

y_feat = y[feat_cols].copy()

# keep only the dates/tickers you need (so history NAs don't matter)
keys = df[["ticker","date_"]].drop_duplicates()
y_feat = keys.merge(y_feat, on=["ticker","date_"], how="left")

# merge into df without renaming df
df = df.merge(y_feat, on=["ticker","date_"], how="left")

# optional: check missing features
feature_cols_only = [c for c in feat_cols if c not in ["ticker","date_"]]
missing = df[feature_cols_only].isna().sum().sort_values(ascending=False)
missing[missing > 0]

# --- date bounds for the ML panel ---
START = pd.Timestamp("2018-01-01")
END   = pd.Timestamp("2024-12-31")

# --- clean yahoo_hist ---
y = yahoo_hist.copy()
y["date_"] = pd.to_datetime(y["date_"], errors="coerce").dt.normalize()
y = y.dropna(subset=["ticker", "date_", "close_", "open_", "high", "low", "volume"])
y = y.sort_values(["ticker", "date_"]).reset_index(drop=True)

# keep only panel window, but include a little extra history BEFORE START for lags/rolls
# (we’ll filter to START..END at the end)
y = y[y["date_"] <= END].copy()

g = y.groupby("ticker")

# --- Returns (log), using only past prices ---
y["ret_1"]   = np.log(y["close_"] / g["close_"].shift(1))
for k in [2,5,10,22,60,90,120,150,180,200]:
    # return over last k trading days ending at t-1
    y[f"ret_{k}"] = np.log(g["close_"].shift(1) / g["close_"].shift(1 + k))

# --- Range / intraday movement ---
y["hl_log"] = np.log(y["high"] / y["low"])
y["oc_log"] = np.log(y["close_"] / y["open_"])
y["co_log"] = np.log(y["open_"] / g["close_"].shift(1))

# --- Volume ---
vol = y["volume"].astype(float)
y["log_vol"]  = np.log(vol.where(vol > 0))
y["dlog_vol"] = g["log_vol"].diff()

# --- Rolling vol proxies (past), using info up to t-1 ---
y["rv1"] = y["ret_1"] ** 2
for w in [5, 22]:
    y[f"rv{w}"]  = g["rv1"].transform(lambda s: s.shift(1).rolling(w).sum())
    y[f"vol{w}"] = np.sqrt(y[f"rv{w}"])

# --- Range-based rolling proxies (past) ---
for w in [5, 22]:
    y[f"hl_{w}"] = g["hl_log"].transform(lambda s: s.shift(1).rolling(w).mean())

# --- Momentum / trend (optional) ---
y["mom_5"]   = y["ret_5"]
y["mom_22"]  = y["ret_22"]
y["mom_200"] = y["ret_200"]

# --- Forward realized volatility targets from future returns (NO leakage in predictors) ---
y["ret_fwd_1"] = np.log(g["close_"].shift(-1) / y["close_"])
y["sq_ret_fwd_1"] = y["ret_fwd_1"] ** 2

y["rv_fwd_3"]  = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(3).sum())
y["rv_fwd_5"]  = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(5).sum())
y["rv_fwd_10"] = g["sq_ret_fwd_1"].transform(lambda s: s.rolling(10).sum())

y["vol_fwd_3"]  = np.sqrt(y["rv_fwd_3"])
y["vol_fwd_5"]  = np.sqrt(y["rv_fwd_5"])
y["vol_fwd_10"] = np.sqrt(y["rv_fwd_10"])

# --- final panel: filter to START..END and select columns ---
y = y[(y["date_"] >= START) & (y["date_"] <= END)].copy()

# columns to keep (identifiers + targets + features)
id_cols = ["ticker", "date_"]
target_cols = ["vol_fwd_3", "vol_fwd_5", "vol_fwd_10"]
feature_cols = [
    "ret_1","ret_2","ret_5","ret_10","ret_22","ret_60","ret_90","ret_120","ret_150","ret_180","ret_200",
    "hl_log","oc_log","co_log",
    "log_vol","dlog_vol",
    "rv1","rv5","rv22","vol5","vol22",
    "hl_5","hl_22",
    "mom_5","mom_22","mom_200",
]

# build df (THIS overwrites df with the Yahoo panel for ML)
df = y[id_cols + target_cols + feature_cols].copy()

# drop rows missing targets (can’t train/evaluate without them)
df = df.dropna(subset=target_cols).reset_index(drop=True)

# drop rows missing longest lookback available
need = ["ret_200", "rv22", "vol22", "hl_22", "mom_200"]  # covers longest deps
df = df.dropna(subset=need).reset_index(drop=True)

df.head(), df.shape

# df_predictions: one row per (ticker, date_) with realized future vol targets
# Assumes df already contains: date_, ticker, vol_fwd_3, vol_fwd_5, vol_fwd_10

df_predictions = (
    df[["date_", "ticker", "vol_fwd_3", "vol_fwd_5", "vol_fwd_10"]]
    .copy()
    .sort_values(["ticker", "date_"])
    .reset_index(drop=True)
)

import pandas as pd
import numpy as np

# metrics container: one row per (model, horizon, fold)
results_cols = [
    "model",          # e.g., "GARCH(1,1)-t"
    "horizon",        # 3, 5, 10
    "fold_id",        # integer
    "train_start",    # date
    "train_end",      # date (last train date)
    "test_start",     # date
    "test_end",       # date
    "n_train",
    "n_test",
    "mse",
    "rmse",
    "mae",
    "mape",           # optional; can be NaN if you skip it
]

df_results = pd.DataFrame(columns=results_cols)

# ---- metric helpers (use on aligned arrays/Series) ----
def _to_float_array(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean((y_true[m] - y_pred[m])**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))

def mape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    denom = np.maximum(np.abs(y_true[m]), eps)
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / denom)))

# ---- helper to append one result row ----
def add_result_row(
    df_results,
    model, horizon, fold_id,
    train_start, train_end, test_start, test_end,
    y_true, y_pred
):
    row = {
        "model": model,
        "horizon": int(horizon),
        "fold_id": int(fold_id),
        "train_start": pd.to_datetime(train_start),
        "train_end": pd.to_datetime(train_end),
        "test_start": pd.to_datetime(test_start),
        "test_end": pd.to_datetime(test_end),
        "n_train": int(np.sum(~np.isnan(np.asarray(y_true, dtype=float)))) ,  # placeholder; set properly when you have splits
        "n_test": int(np.sum(~np.isnan(np.asarray(y_true, dtype=float)))) ,   # placeholder; set properly when you have splits
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    return pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

######### ALGORITHM

# ============================================================
# CONFIGURATION  (overridable via environment variables)
# ============================================================
import os as _os

TRAIN_END    = pd.Timestamp("2023-11-30")
DEC_START    = pd.Timestamp("2023-12-01")
DEC_END      = pd.Timestamp("2023-12-31")
HORIZONS     = [3, 5, 10]
TRADING_DAYS = 252
THRESHOLD_LONG        = float(_os.environ.get("THRESHOLD_LONG",   "-0.01"))
THRESHOLD_SHORT       = float(_os.environ.get("THRESHOLD_SHORT",   "0.0"))
BID_ASK_SPREAD_IMPUTE = 0.03
LIQUIDITY_FILTER      = 0.6
CAPITAL_PER_TRADE     = float(_os.environ.get("CAPITAL_PER_TRADE", "200"))
INITIAL_CAPITAL       = float(_os.environ.get("INITIAL_CAPITAL",   "10000"))
PLOWBACK_RATIO        = float(_os.environ.get("PLOWBACK_RATIO",    "0.0"))

# Dynamic MIN_DTE per horizon (base values + user delta)
_min_dte_delta = int(_os.environ.get("MIN_DTE_DELTA", "0"))
MIN_DTE = {3: 12 + _min_dte_delta, 5: 14 + _min_dte_delta, 10: 18 + _min_dte_delta}

# ============================================================
# HELPERS
# ============================================================
def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    return df

def ann_to_hday(sigma_ann, h, td=TRADING_DAYS):
    return sigma_ann * np.sqrt(h / td)

def best_quotes(op_n, spread=BID_ASK_SPREAD_IMPUTE):
    q = op_n.groupby(["optid", "date_"], as_index=False).agg(
        best_bid=("bid", "max"), best_ask=("ask", "min")
    )
    m_bid = q["best_bid"].isna() & q["best_ask"].notna()
    m_ask = q["best_ask"].isna() & q["best_bid"].notna()
    q.loc[m_bid, "best_bid"] = q.loc[m_bid, "best_ask"] * (1 - spread)
    q.loc[m_ask, "best_ask"] = q.loc[m_ask, "best_bid"] * (1 + spread)
    return q

def select_atm_straddle(base, oc, min_dte=0):
    oc = oc[oc["putcall"].isin(["C","P"])].dropna(
        subset=["optid","opraticker","strike","expdate","multiplier"]
    ).copy()
    m = base.merge(oc, left_on="ticker", right_on="opraticker", how="left")
    m = m.dropna(subset=["optid","strike","expdate","multiplier","exit_date"])
    m = m[m["expdate"] > m["exit_date"]].copy()
    m = m[((m["expdate"] - m["date_"]).dt.days) >= min_dte].copy()
    if m.empty:
        return pd.DataFrame(columns=[
            "ticker","date_","optid_call","optid_put",
            "strike_call","strike_put","expdate_selected","multiplier"
        ])
    m["min_exp"] = m.groupby(["ticker","date_"])["expdate"].transform("min")
    m = m[m["expdate"] == m["min_exp"]].copy()
    m["abs_mny"] = (m["strike"] - m["spot"]).abs()
    m["mny_rank"] = m.groupby(["ticker","date_","putcall"])["abs_mny"].rank(method="first")
    m = m[m["mny_rank"] == 1].copy()
    piv_id  = m.pivot_table(index=["ticker","date_"], columns="putcall", values="optid",      aggfunc="first").rename(columns={"C":"optid_call",  "P":"optid_put"})
    piv_str = m.pivot_table(index=["ticker","date_"], columns="putcall", values="strike",     aggfunc="first").rename(columns={"C":"strike_call", "P":"strike_put"})
    piv_exp = m.pivot_table(index=["ticker","date_"], columns="putcall", values="expdate",    aggfunc="first").rename(columns={"C":"expdate_C",   "P":"expdate_P"})
    piv_mul = m.pivot_table(index=["ticker","date_"], columns="putcall", values="multiplier", aggfunc="first").rename(columns={"C":"mult_C",      "P":"mult_P"})
    out = piv_id.join([piv_str, piv_exp, piv_mul]).reset_index()
    out["expdate_selected"] = out[["expdate_C","expdate_P"]].min(axis=1)
    out["multiplier"] = np.where(out["mult_C"] == out["mult_P"], out["mult_C"], np.nan)
    return out[["ticker","date_","optid_call","optid_put",
                "strike_call","strike_put","expdate_selected","multiplier"]]

def attach_quotes(df, bestq, date_col, suffix):
    bq = bestq.rename(columns={"date_": date_col})
    df = df.merge(
        bq.rename(columns={"best_bid": f"call_bid{suffix}", "best_ask": f"call_ask{suffix}",
                            "optid": "optid_call"}),
        on=["optid_call", date_col], how="left"
    )
    df = df.merge(
        bq.rename(columns={"best_bid": f"put_bid{suffix}", "best_ask": f"put_ask{suffix}",
                            "optid": "optid_put"}),
        on=["optid_put", date_col], how="left"
    )
    return df

def straddle_value(df, kind, suffix, mult_col="multiplier"):
    c = df[f"call_{kind}{suffix}"]
    p = df[f"put_{kind}{suffix}"]
    return (c + p) * df[mult_col]

# ============================================================
# STEP 1 — LASSO-PER-TICKER FORECASTS FOR DECEMBER 2023
# ============================================================
from sklearn.linear_model import Lasso

_lasso_df = df.copy()
_lasso_df["date_"] = pd.to_datetime(_lasso_df["date_"], errors="coerce").dt.normalize()
_lasso_df = (_lasso_df.dropna(subset=["ticker","date_"])
             .sort_values(["ticker","date_"]).reset_index(drop=True))

_target_cols  = [f"vol_fwd_{h}" for h in HORIZONS]
_id_cols      = ["ticker", "date_"]
_feature_cols = [c for c in _lasso_df.columns if c not in set(_id_cols + _target_cols)]
_lasso_df[_feature_cols] = _lasso_df[_feature_cols].apply(pd.to_numeric, errors="coerce")

_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

def _pick_alpha_lasso(X, y, alphas):
    n = len(X)
    if n < 300:
        return 1e-2
    cut1, cut2 = int(n * 0.6), int(n * 0.8)
    splits = [(slice(0, cut1), slice(cut1, cut2)), (slice(0, cut2), slice(cut2, n))]
    best_a, best_score = None, np.inf
    for a in alphas:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("lasso", Lasso(alpha=a, max_iter=20000, random_state=123))])
        scores = []
        for tr, va in splits:
            Xtr, ytr = X.iloc[tr], y[tr]
            Xva, yva = X.iloc[va], y[va]
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            yt, yp = np.asarray(yva, float), np.asarray(pred, float)
            m = np.isfinite(yt) & np.isfinite(yp)
            sc = float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2))) if m.any() else np.nan
            scores.append(sc)
        sc = np.nanmean(scores)
        if np.isfinite(sc) and sc < best_score:
            best_score = sc
            best_a = a
    return float(best_a) if best_a is not None else 1e-2

pred_rows = []
for tkr, d_tkr in _lasso_df.groupby("ticker"):
    d_tkr   = d_tkr.sort_values("date_").reset_index(drop=True)
    d_train = d_tkr[d_tkr["date_"] <= TRAIN_END].copy()
    d_dec   = d_tkr[(d_tkr["date_"] >= DEC_START) & (d_tkr["date_"] <= DEC_END)].copy()
    if len(d_train) < 252 or d_dec.empty:
        continue
    row = pd.DataFrame({"ticker": tkr, "date_": d_dec["date_"].values})
    for h in HORIZONS:
        ycol    = f"vol_fwd_{h}"
        d_tr    = d_train.dropna(subset=[ycol])
        if len(d_tr) < 252:
            continue
        X_train = d_tr[_feature_cols].copy()
        X_dec   = d_dec[_feature_cols].copy()
        med     = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_dec   = X_dec.fillna(med)
        y_train = d_tr[ycol].astype(float).values
        alpha_star = _pick_alpha_lasso(X_train, y_train, _ALPHA_GRID)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("lasso", Lasso(alpha=alpha_star, max_iter=20000, random_state=123))])
        pipe.fit(X_train, y_train)
        row[f"rfwd_tplus{h}"] = pipe.predict(X_dec)
    pred_rows.append(row)

df_lasso_dec   = pd.concat(pred_rows, ignore_index=True)
df_figarch_dec = df_lasso_dec   # alias so downstream steps remain unchanged
print(f"Lasso forecasts: {len(df_lasso_dec)} rows, {df_lasso_dec['ticker'].nunique()} tickers")

# ============================================================
# STEP 2 — BUILD MASTER STRATEGY DATAFRAME
# ============================================================
oc_n = norm_cols(option_contracts)
op_n = norm_cols(option_prices)
iv_n = norm_cols(iv)

oc_n["expdate"] = pd.to_datetime(oc_n["expdate"], errors="coerce").dt.normalize()
op_n["date_"]   = pd.to_datetime(op_n["date_"],   errors="coerce").dt.normalize()
iv_n["date_"]   = pd.to_datetime(iv_n["date_"],   errors="coerce").dt.normalize()
oc_n["putcall"]  = oc_n["putcall"].astype(str).str.upper().str[0]

bestq  = best_quotes(op_n)
y_spot = (yahoo_hist[["ticker","date_","close_"]]
          .assign(date_=lambda x: pd.to_datetime(x["date_"], errors="coerce").dt.normalize())
          .dropna().sort_values(["ticker","date_"]))

cal = (y_spot[["ticker","date_"]].drop_duplicates()
       .sort_values(["ticker","date_"]).reset_index(drop=True))
for h in HORIZONS:
    cal[f"exit_date_{h}"] = cal.groupby("ticker")["date_"].shift(-h)

strat_df = df_figarch_dec.copy()
strat_df["date_"] = pd.to_datetime(strat_df["date_"], errors="coerce").dt.normalize()
strat_df = strat_df.merge(y_spot.rename(columns={"close_":"spot"}), on=["ticker","date_"], how="left")
strat_df = strat_df.merge(cal, on=["ticker","date_"], how="left")

# ============================================================
# STEP 3 — FOR EACH HORIZON: STRADDLE SELECTION + IV + QUOTES
# ============================================================
for h in HORIZONS:
    exit_col = f"exit_date_{h}"

    base = (strat_df[["ticker","date_","spot", exit_col]]
            .rename(columns={exit_col: "exit_date"})
            .dropna(subset=["spot","exit_date"]))

    sel = select_atm_straddle(base, oc_n, min_dte=MIN_DTE[h])

    for col in ["optid_call","optid_put","strike_call","strike_put","expdate_selected","multiplier"]:
        strat_df = strat_df.merge(
            sel[["ticker","date_",col]].rename(columns={col: f"{col}_{h}"}),
            on=["ticker","date_"], how="left"
        )

    # ── IV WITH NEAREST-DATE FALLBACK ─────────────────────────
    iv_use = iv_n[["optid","date_","ivmid"]].dropna(subset=["ivmid"])

    sel_iv = sel.merge(
        iv_use.rename(columns={"optid":"optid_call","ivmid":"iv_call"}),
        on=["optid_call","date_"], how="left"
    ).merge(
        iv_use.rename(columns={"optid":"optid_put","ivmid":"iv_put"}),
        on=["optid_put","date_"], how="left"
    )

    missing_call = sel_iv["iv_call"].isna()
    missing_put  = sel_iv["iv_put"].isna()

    if missing_call.any():
        for idx, row in sel_iv[missing_call].iterrows():
            nearby = iv_use[iv_use["optid"] == row["optid_call"]].sort_values("date_")
            if not nearby.empty:
                diffs = (nearby["date_"] - row["date_"]).abs()
                sel_iv.loc[idx, "iv_call"] = nearby.loc[diffs.idxmin(), "ivmid"]

    if missing_put.any():
        for idx, row in sel_iv[missing_put].iterrows():
            nearby = iv_use[iv_use["optid"] == row["optid_put"]].sort_values("date_")
            if not nearby.empty:
                diffs = (nearby["date_"] - row["date_"]).abs()
                sel_iv.loc[idx, "iv_put"] = nearby.loc[diffs.idxmin(), "ivmid"]

    sel_iv[f"iv_ann_{h}"]  = sel_iv[["iv_call","iv_put"]].mean(axis=1)
    sel_iv[f"iv_hday_{h}"] = ann_to_hday(sel_iv[f"iv_ann_{h}"], h)

    strat_df = strat_df.merge(
        sel_iv[["ticker","date_", f"iv_ann_{h}", f"iv_hday_{h}"]],
        on=["ticker","date_"], how="left"
    )

    # ── ENTRY QUOTES ──────────────────────────────────────────
    sel_q = attach_quotes(sel, bestq, date_col="date_", suffix="_entry")
    sel_q[f"straddle_ask_entry_{h}"] = straddle_value(sel_q, "ask", "_entry")
    sel_q[f"straddle_bid_entry_{h}"] = straddle_value(sel_q, "bid", "_entry")

    strat_df = strat_df.merge(
        sel_q[["ticker","date_", f"straddle_ask_entry_{h}", f"straddle_bid_entry_{h}"]],
        on=["ticker","date_"], how="left"
    )

    # ── EXIT QUOTES ───────────────────────────────────────────
    sel_exit = sel.copy()
    sel_exit["exit_date"] = strat_df.set_index(["ticker","date_"])[exit_col].reindex(
        pd.MultiIndex.from_frame(sel_exit[["ticker","date_"]])
    ).values
    sel_exit = sel_exit.dropna(subset=["exit_date"])
    sel_exit = attach_quotes(sel_exit, bestq, date_col="exit_date", suffix="_exit")
    sel_exit[f"straddle_bid_exit_{h}"] = straddle_value(sel_exit, "bid", "_exit")
    sel_exit[f"straddle_ask_exit_{h}"] = straddle_value(sel_exit, "ask", "_exit")

    strat_df = strat_df.merge(
        sel_exit[["ticker","date_", f"straddle_bid_exit_{h}", f"straddle_ask_exit_{h}"]],
        on=["ticker","date_"], how="left"
    )

    # ── EXIT QUOTE ROLL ───────────────────────────────────────
    need_roll = (strat_df[f"straddle_bid_exit_{h}"].isna() |
                 strat_df[f"straddle_ask_exit_{h}"].isna())

    if need_roll.any():
        roll_dates = (strat_df.loc[need_roll, ["ticker", exit_col]]
                      .rename(columns={exit_col: "date_"})
                      .dropna().drop_duplicates())
        roll_dates = roll_dates.merge(
            y_spot.rename(columns={"close_":"spot"}), on=["ticker","date_"], how="left"
        ).dropna(subset=["spot"])
        roll_dates["exit_date"] = roll_dates["date_"]
        roll_sel = select_atm_straddle(roll_dates, oc_n, min_dte=0)
        if not roll_sel.empty:
            roll_sel = attach_quotes(roll_sel, bestq, date_col="date_", suffix="_exit")
            roll_sel[f"roll_bid_{h}"] = straddle_value(roll_sel, "bid", "_exit")
            roll_sel[f"roll_ask_{h}"] = straddle_value(roll_sel, "ask", "_exit")
            roll_sel = roll_sel.rename(columns={"date_": exit_col})
            strat_df = strat_df.merge(
                roll_sel[["ticker", exit_col, f"roll_bid_{h}", f"roll_ask_{h}"]],
                on=["ticker", exit_col], how="left"
            )
            strat_df.loc[need_roll, f"straddle_bid_exit_{h}"] = strat_df.loc[need_roll, f"roll_bid_{h}"]
            strat_df.loc[need_roll, f"straddle_ask_exit_{h}"] = strat_df.loc[need_roll, f"roll_ask_{h}"]
            strat_df.drop(columns=[f"roll_bid_{h}", f"roll_ask_{h}"], inplace=True)

    strat_df[f"exit_rolled_{h}"] = need_roll

# ============================================================
# STEP 4 — SIGNAL
# ============================================================
for h in HORIZONS:
    strat_df[f"vol_spread_{h}"] = strat_df[f"rfwd_tplus{h}"] - strat_df[f"iv_hday_{h}"]
    strat_df[f"signal_{h}"] = np.where(
        strat_df[f"vol_spread_{h}"] >  THRESHOLD_LONG,   1,
        np.where(
        strat_df[f"vol_spread_{h}"] < -THRESHOLD_SHORT, -1,
        0)
    )

# ============================================================
# STEP 5 — TRADE-LEVEL P&L WITH EQUAL CAPITAL SIZING
# ============================================================
for h in HORIZONS:
    ask_e = f"straddle_ask_entry_{h}"
    bid_e = f"straddle_bid_entry_{h}"
    bid_x = f"straddle_bid_exit_{h}"
    ask_x = f"straddle_ask_exit_{h}"
    sig   = f"signal_{h}"

    long_quotes  = strat_df[[ask_e, bid_x]].notna().all(axis=1)
    short_quotes = strat_df[[bid_e, ask_x]].notna().all(axis=1)

    liquid_long  = long_quotes  & ((strat_df[bid_x] / strat_df[ask_e]) >= LIQUIDITY_FILTER)
    liquid_short = short_quotes & ((strat_df[ask_x] / strat_df[bid_e]) >= LIQUIDITY_FILTER)

    long_ok  = (strat_df[sig] ==  1) & liquid_long
    short_ok = (strat_df[sig] == -1) & liquid_short

    n_contracts = pd.Series(np.nan, index=strat_df.index)
    n_contracts[long_ok]  = CAPITAL_PER_TRADE / strat_df.loc[long_ok,  ask_e]
    n_contracts[short_ok] = CAPITAL_PER_TRADE / strat_df.loc[short_ok, bid_e]
    strat_df[f"n_contracts_{h}"] = n_contracts

    entry_cashflow = pd.Series(0.0, index=strat_df.index)
    entry_cashflow[long_ok]  = -CAPITAL_PER_TRADE
    entry_cashflow[short_ok] = +CAPITAL_PER_TRADE
    strat_df[f"entry_cashflow_{h}"] = entry_cashflow

    exit_cashflow = pd.Series(0.0, index=strat_df.index)
    exit_cashflow[long_ok]  = +(strat_df.loc[long_ok,  bid_x] * n_contracts[long_ok])
    exit_cashflow[short_ok] = -(strat_df.loc[short_ok, ask_x] * n_contracts[short_ok])
    strat_df[f"exit_cashflow_{h}"] = exit_cashflow

    pnl = pd.Series(np.nan, index=strat_df.index)
    pnl[long_ok]  = entry_cashflow[long_ok]  + exit_cashflow[long_ok]
    pnl[short_ok] = entry_cashflow[short_ok] + exit_cashflow[short_ok]
    strat_df[f"pnl_{h}"] = pnl

    ret = pd.Series(np.nan, index=strat_df.index)
    ret[long_ok | short_ok] = pnl[long_ok | short_ok] / CAPITAL_PER_TRADE
    strat_df[f"trade_ret_{h}"] = ret

    strat_df[f"liquidity_filtered_{h}"] = (
        ((strat_df[sig] ==  1) & long_quotes  & ~liquid_long) |
        ((strat_df[sig] == -1) & short_quotes & ~liquid_short)
    )

# ============================================================
# STEP 6 — DAILY PORTFOLIO SIMULATION
# ============================================================
all_daily_summaries = {}

for h in HORIZONS:
    sig_col        = f"signal_{h}"
    ret_col        = f"trade_ret_{h}"
    vol_spread_col = f"vol_spread_{h}"
    exit_col       = f"exit_date_{h}"

    tradeable = strat_df[
        (strat_df[sig_col] != 0) & strat_df[ret_col].notna()
    ].copy()

    entry_dates = set(pd.to_datetime(tradeable["date_"].unique()))
    exit_dates  = set(pd.to_datetime(tradeable[exit_col].dropna().unique()))
    _dec_days   = set(pd.to_datetime(
        yahoo_hist[(yahoo_hist["date_"] >= DEC_START) &
                   (yahoo_hist["date_"] <= DEC_END)]["date_"]
        .dt.normalize().unique()
    ))
    all_dates   = sorted(entry_dates | exit_dates | _dec_days)

    # portfolio state
    cash                 = float(INITIAL_CAPITAL)
    long_positions_value = 0.0
    short_liability      = 0.0
    gains_not_reinvested = 0.0
    pending_cash         = 0.0

    open_positions = {}
    executed_idx   = set()
    daily_rows     = []

    for date in all_dates:
        date = pd.Timestamp(date)

        # gains reinvested today = pending cash from yesterday
        daily_gains_reinvested = pending_cash
        cash        += pending_cash
        pending_cash = 0.0

        # ── BEG STATE ─────────────────────────────────────────
        beg_cash                 = cash
        beg_long_positions_value = long_positions_value
        beg_short_liability      = short_liability
        beg_gains_not_reinvested = gains_not_reinvested
        beg_total_portfolio      = (beg_cash + beg_long_positions_value +
                                    beg_short_liability + beg_gains_not_reinvested)

        # daily accumulators — longs
        long_amount_invested        = 0.0
        long_amount_collected       = 0.0
        n_long_opened               = 0
        n_long_closed               = 0
        n_long_profitable_closed    = 0
        n_long_loss_closed          = 0
        long_gains_obtained         = 0.0
        long_gains_not_reinvested   = 0.0
        long_losses                 = 0.0

        # daily accumulators — shorts
        short_amount_collected      = 0.0
        short_amount_invested       = 0.0
        n_short_opened              = 0
        n_short_closed              = 0
        n_short_profitable_closed   = 0
        n_short_loss_closed         = 0
        short_gains_obtained        = 0.0
        short_gains_not_reinvested  = 0.0
        short_losses                = 0.0

        # other
        lack_of_funds   = 0
        open_tickers    = []
        close_tickers   = []

        # ── OPENS ─────────────────────────────────────────────
        opens_today = tradeable[tradeable["date_"] == date].copy()
        opens_today = opens_today.sort_values(vol_spread_col, ascending=False)

        for idx, row in opens_today.iterrows():
            sig = row[sig_col]

            if sig == 1:  # LONG: deploy cash
                if cash >= CAPITAL_PER_TRADE:
                    cash                 -= CAPITAL_PER_TRADE
                    long_positions_value += CAPITAL_PER_TRADE
                    long_amount_invested += CAPITAL_PER_TRADE
                    n_long_opened        += 1
                    executed_idx.add(idx)
                    open_positions[(row["ticker"], date, idx)] = {
                        "signal":     1,
                        "entry_cost": CAPITAL_PER_TRADE
                    }
                    open_tickers.append(row["ticker"])
                else:
                    lack_of_funds += 1

            elif sig == -1:  # SHORT: book liability, no cash movement
                premium                = CAPITAL_PER_TRADE
                short_liability       += premium
                short_amount_collected += premium
                n_short_opened        += 1
                executed_idx.add(idx)
                open_positions[(row["ticker"], date, idx)] = {
                    "signal":     -1,
                    "entry_cost":  premium
                }
                open_tickers.append(row["ticker"])

        # ── CLOSES ────────────────────────────────────────────
        closes_today = tradeable[
            (tradeable[exit_col] == date) &
            (tradeable.index.isin(executed_idx))
        ].copy()

        for idx, row in closes_today.iterrows():
            sig = row[sig_col]

            pos_key = next(
                (k for k in open_positions if k[0] == row["ticker"] and k[2] == idx),
                None
            )
            if pos_key is None:
                continue

            pos        = open_positions.pop(pos_key)
            entry_cost = pos["entry_cost"]

            if sig == 1:  # LONG CLOSE
                proceeds               = row[f"exit_cashflow_{h}"]
                long_positions_value  -= entry_cost
                long_amount_collected += proceeds
                n_long_closed         += 1
                close_tickers.append(row["ticker"])

                if proceeds >= entry_cost:
                    # recover capital to cash + book gain via plowback
                    cash += entry_cost
                    gain  = proceeds - entry_cost
                    long_gains_obtained      += gain
                    n_long_profitable_closed += 1
                    if PLOWBACK_RATIO > 0:
                        long_gains_not_reinvested += gain * (1 - PLOWBACK_RATIO)
                    else:
                        long_gains_not_reinvested += gain
                else:
                    # partial recovery only
                    loss  = entry_cost - proceeds
                    cash += proceeds
                    long_losses        += loss
                    n_long_loss_closed += 1

            elif sig == -1:  # SHORT CLOSE
                close_cost             = abs(row[f"exit_cashflow_{h}"])
                short_liability       -= entry_cost
                short_amount_invested += close_cost
                n_short_closed        += 1
                close_tickers.append(row["ticker"])

                if close_cost <= entry_cost:
                    # profitable: no cash movement, book gain via plowback
                    gain = entry_cost - close_cost
                    short_gains_obtained      += gain
                    n_short_profitable_closed += 1
                    if PLOWBACK_RATIO > 0:
                        short_gains_not_reinvested += gain * (1 - PLOWBACK_RATIO)
                    else:
                        short_gains_not_reinvested += gain
                else:
                    # loss: pay out of pocket
                    loss = close_cost - entry_cost
                    cash -= loss
                    short_losses        += loss
                    n_short_loss_closed += 1

        # ── PLOWBACK ──────────────────────────────────────────
        total_gains_obtained  = long_gains_obtained + short_gains_obtained
        total_gains_not_reinv = long_gains_not_reinvested + short_gains_not_reinvested
        gains_not_reinvested += total_gains_not_reinv
        pending_cash          = total_gains_obtained * PLOWBACK_RATIO

        # ── END STATE ─────────────────────────────────────────
        end_cash                 = cash
        end_long_positions_value = long_positions_value
        end_short_liability      = short_liability
        end_gains_not_reinvested = gains_not_reinvested
        end_total_portfolio      = (end_cash + end_long_positions_value +
                                    end_short_liability + end_gains_not_reinvested +
                                    pending_cash)

        daily_rows.append({
            # BEG STATE
            "date":                             date,
            "horizon":                          h,
            "beg_cash":                         round(beg_cash, 2),
            "beg_long_positions_value":         round(beg_long_positions_value, 2),
            "beg_short_liability":              round(beg_short_liability, 2),
            "beg_gains_not_reinvested":         round(beg_gains_not_reinvested, 2),
            "beg_total_portfolio_value":        round(beg_total_portfolio, 2),
            # LONG ACTIVITY
            "n_long_opened":                    n_long_opened,
            "long_amount_invested":             round(long_amount_invested, 2),
            "n_long_closed":                    n_long_closed,
            "n_long_profitable_closed":         n_long_profitable_closed,
            "n_long_loss_closed":               n_long_loss_closed,
            "long_amount_collected":            round(long_amount_collected, 2),
            "long_gains_obtained":              round(long_gains_obtained, 2),
            "long_gains_not_reinvested":        round(long_gains_not_reinvested, 2),
            "long_losses":                      round(long_losses, 2),
            # SHORT ACTIVITY
            "n_short_opened":                   n_short_opened,
            "short_amount_collected":           round(short_amount_collected, 2),
            "n_short_closed":                   n_short_closed,
            "n_short_profitable_closed":        n_short_profitable_closed,
            "n_short_loss_closed":              n_short_loss_closed,
            "short_amount_invested":            round(short_amount_invested, 2),
            "short_gains_obtained":             round(short_gains_obtained, 2),
            "short_gains_not_reinvested":       round(short_gains_not_reinvested, 2),
            "short_losses":                     round(short_losses, 2),
            # GENERAL ACTIVITY
            "lack_of_funds_trades":             lack_of_funds,
            "tickers_opened":                   ", ".join(open_tickers)  if open_tickers  else "",
            "tickers_closed":                   ", ".join(close_tickers) if close_tickers else "",
            "daily_gains_reinvested":           round(daily_gains_reinvested, 2),
            "daily_gains_obtained":             round(total_gains_obtained, 2),
            "daily_gains_not_reinvested":       round(total_gains_not_reinv, 2),
            "daily_losses":                     round(long_losses + short_losses, 2),
            # END STATE
            "end_cash":                         round(end_cash, 2),
            "end_long_positions_value":         round(end_long_positions_value, 2),
            "end_short_liability":              round(end_short_liability, 2),
            "end_gains_not_reinvested":         round(end_gains_not_reinvested, 2),
            "end_total_portfolio_value":        round(end_total_portfolio, 2),
            "unleveraged_total_portfolio_value": round(end_total_portfolio - end_short_liability, 2),
        })

    all_daily_summaries[h] = pd.DataFrame(daily_rows)

# ============================================================
# STEP 7 — DAILY SUMMARY TABLE
# ============================================================
daily_summary_rows = []

for h in HORIZONS:
    ds         = all_daily_summaries[h]
    all_dates_h = sorted(pd.to_datetime(ds["date"]).unique())

    for current_date in all_dates_h:
        current_date = pd.Timestamp(current_date)
        ds_up = ds[pd.to_datetime(ds["date"]) <= current_date]

        total_long_invested    = ds_up["long_amount_invested"].sum()
        total_long_collected   = ds_up["long_amount_collected"].sum()
        total_short_collected  = ds_up["short_amount_collected"].sum()
        total_short_invested   = ds_up["short_amount_invested"].sum()
        total_invested         = total_long_invested  + total_short_invested
        total_collected        = total_long_collected + total_short_collected
        total_gains_obtained   = ds_up["daily_gains_obtained"].sum()
        total_gains_not_reinv  = ds_up["daily_gains_not_reinvested"].sum()
        total_gains_reinvested = ds_up["daily_gains_reinvested"].sum()
        total_long_losses      = ds_up["long_losses"].sum()
        total_short_losses     = ds_up["short_losses"].sum()
        total_losses           = ds_up["daily_losses"].sum()
        unleveraged_final      = ds_up["unleveraged_total_portfolio_value"].iloc[-1]
        cash_on_cash_ret       = (total_collected / total_invested - 1) if total_invested > 0 else np.nan
        simple_ret             = (unleveraged_final / INITIAL_CAPITAL) - 1

        port_series = ds_up["unleveraged_total_portfolio_value"]
        rolling_max = port_series.cummax()
        drawdown    = (port_series - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        daily_summary_rows.append({
            "date":                         current_date,
            "horizon":                      h,
            # capital & returns
            "initial_capital":              INITIAL_CAPITAL,
            "final_portfolio_value":        round(unleveraged_final, 2),
            "simple_ret_%":                 round(100 * simple_ret, 2),
            "cash_on_cash_ret_%":           round(100 * cash_on_cash_ret, 2),
            "max_drawdown_%":               round(100 * max_dd, 2),
            # long activity
            "total_long_invested":          round(total_long_invested, 2),
            "total_long_collected":         round(total_long_collected, 2),
            "total_long_gains":             round(total_gains_obtained - ds_up["short_gains_obtained"].sum(), 2),
            "total_long_losses":            round(total_long_losses, 2),
            "n_long_opened":                int(ds_up["n_long_opened"].sum()),
            "n_long_profitable_closed":     int(ds_up["n_long_profitable_closed"].sum()),
            "n_long_loss_closed":           int(ds_up["n_long_loss_closed"].sum()),
            # short activity
            "total_short_collected":        round(total_short_collected, 2),
            "total_short_invested":         round(total_short_invested, 2),
            "total_short_gains":            round(ds_up["short_gains_obtained"].sum(), 2),
            "total_short_losses":           round(total_short_losses, 2),
            "n_short_opened":               int(ds_up["n_short_opened"].sum()),
            "n_short_profitable_closed":    int(ds_up["n_short_profitable_closed"].sum()),
            "n_short_loss_closed":          int(ds_up["n_short_loss_closed"].sum()),
            # general
            "total_gains_obtained":         round(total_gains_obtained, 2),
            "total_gains_not_reinvested":   round(total_gains_not_reinv, 2),
            "total_gains_reinvested":       round(total_gains_reinvested, 2),
            "total_losses":                 round(total_losses, 2),
            "n_trade_dates":                int((ds_up["n_long_opened"] + ds_up["n_short_opened"] > 0).sum()),
            "lack_of_funds_trades":         int(ds_up["lack_of_funds_trades"].sum()),
        })

summary_df = pd.DataFrame(daily_summary_rows)

def show_day_transactions(date_str, h=10):
    date     = pd.Timestamp(date_str)
    exit_col = f"exit_date_{h}"
    sig_col  = f"signal_{h}"
    ret_col  = f"trade_ret_{h}"

    tradeable = strat_df[(strat_df[sig_col] != 0) & strat_df[ret_col].notna()].copy()

    # trades opened on this date
    opens = tradeable[tradeable["date_"] == date].copy()
    opens["transaction"] = "OPEN"
    opens["entry_cost"]  = 1000.0
    opens["exit_cf"]     = np.nan
    opens["pnl"]         = np.nan

    # trades closed on this date
    closes = tradeable[tradeable[exit_col] == date].copy()
    closes["transaction"] = "CLOSE"
    closes["entry_cost"]  = 1000.0
    closes["exit_cf"]     = closes[f"exit_cashflow_{h}"]
    closes["pnl"]         = closes[f"pnl_{h}"]

    combined = pd.concat([opens, closes]).sort_values(["transaction", "ticker"])

    cols = [
        "transaction", "ticker", "date_", exit_col, sig_col,
        f"straddle_ask_entry_{h}", f"straddle_bid_exit_{h}",
        "entry_cost", "exit_cf", "pnl", ret_col
    ]

    print(f"\n=== Transactions for {date_str} (h={h}) ===")
    print(f"Opens:  {len(opens)} | Closes: {len(closes)}")
    display(combined[cols].reset_index(drop=True))

# ============================================================
# STEP 8 — DAILY PERFORMANCE METRICS
# ============================================================
RISK_FREE_RATE = 0.02                      # annual
rfr_period     = RISK_FREE_RATE / 12       # scaled to one calendar month (December)
rfr_daily      = RISK_FREE_RATE / TRADING_DAYS  # per trading day, for annualised Sharpe

# ── Pre-compute per-trade benchmark returns (done once, reused across dates) ──
_y_bench = yahoo_hist.copy()
_y_bench["date_"] = pd.to_datetime(_y_bench["date_"], errors="coerce").dt.normalize()
_y_bench = _y_bench.sort_values(["ticker", "date_"])

for h in HORIZONS:
    ret_col  = f"trade_ret_{h}"
    exit_col = f"exit_date_{h}"
    trades_all = strat_df[strat_df[ret_col].notna()].copy()
    bm_rets = []
    for _, row in trades_all.iterrows():
        tkr        = row["ticker"]
        entry_date = row["date_"]
        exit_date  = row[exit_col]
        if pd.isna(exit_date):
            bm_rets.append(np.nan)
            continue
        prices = (_y_bench[(_y_bench["ticker"] == tkr) &
                            (_y_bench["date_"] >= entry_date) &
                            (_y_bench["date_"] <= exit_date)]
                  .sort_values("date_"))
        bm_rets.append((prices["close_"].iloc[-1] / prices["close_"].iloc[0]) - 1
                       if len(prices) >= 2 else np.nan)
    strat_df.loc[trades_all.index, f"bm_ret_{h}"] = bm_rets

# ── Pre-compute buy-and-hold returns (constant across all dates) ──
_bh_all = {}
for tkr in strat_df["ticker"].unique():
    prices = (_y_bench[(_y_bench["ticker"] == tkr) &
                       (_y_bench["date_"] >= DEC_START) &
                       (_y_bench["date_"] <= DEC_END)]
              .sort_values("date_"))
    if len(prices) >= 2:
        _bh_all[tkr] = (prices["close_"].iloc[-1] / prices["close_"].iloc[0]) - 1

buy_hold_ret_all = np.mean(list(_bh_all.values())) if _bh_all else np.nan

_bh_long_by_h = {}
for h in HORIZONS:
    sig_col      = f"signal_{h}"
    ret_col      = f"trade_ret_{h}"
    long_tickers = strat_df[strat_df[ret_col].notna() & (strat_df[sig_col] == 1)]["ticker"].unique()
    vals         = [_bh_all[t] for t in long_tickers if t in _bh_all]
    _bh_long_by_h[h] = np.mean(vals) if vals else np.nan

# ── Daily loop ────────────────────────────────────────────────────────────────
daily_metrics_rows = []

for h in HORIZONS:
    sig_col  = f"signal_{h}"
    ret_col  = f"trade_ret_{h}"
    exit_col = f"exit_date_{h}"
    bm_col   = f"bm_ret_{h}"

    ds          = all_daily_summaries[h]
    all_dates_h = sorted(pd.to_datetime(ds["date"]).unique())

    buy_hold_ret_long = _bh_long_by_h[h]

    for current_date in all_dates_h:
        current_date = pd.Timestamp(current_date)

        # only trades fully closed by current_date
        closed = strat_df[
            strat_df[ret_col].notna() &
            (pd.to_datetime(strat_df[exit_col]) <= current_date)
        ].copy()

        if len(closed) == 0:
            continue

        trade_rets    = closed[ret_col]
        winning       = trade_rets[trade_rets > 0]
        losing        = trade_rets[trade_rets <= 0]
        avg_trade_ret = trade_rets.mean()
        med_trade_ret = trade_rets.median()
        win_rate      = len(winning) / len(trade_rets)
        avg_win       = winning.mean() if len(winning) > 0 else np.nan
        avg_loss      = losing.mean()  if len(losing)  > 0 else np.nan
        vol_returns   = trade_rets.std()

        ds_up             = ds[pd.to_datetime(ds["date"]) <= current_date]
        unleveraged_final = ds_up["unleveraged_total_portfolio_value"].iloc[-1]
        simple_ret        = (unleveraged_final / INITIAL_CAPITAL) - 1
        total_invested    = ds_up["long_amount_invested"].sum() + ds_up["short_amount_invested"].sum()
        total_collected   = ds_up["long_amount_collected"].sum() + ds_up["short_amount_collected"].sum()
        cash_on_cash_ret  = (total_collected / total_invested - 1) if total_invested > 0 else np.nan

        _daily_port_rets = ds_up["unleveraged_total_portfolio_value"].pct_change().dropna()
        _vol_daily       = _daily_port_rets.std()
        sharpe = ((_daily_port_rets.mean() - rfr_daily) / _vol_daily * np.sqrt(TRADING_DAYS)
                  if len(_daily_port_rets) > 1 and _vol_daily > 0 else np.nan)

        trades_clean = closed.dropna(subset=[bm_col])
        if len(trades_clean) > 1:
            cov_matrix = np.cov(trades_clean[ret_col], trades_clean[bm_col])
            beta       = cov_matrix[0, 1] / cov_matrix[1, 1]
        else:
            beta = np.nan

        if not np.isnan(beta):
            _rfr_trade      = RISK_FREE_RATE * h / TRADING_DAYS
            _tc             = trades_clean.copy()
            _tc["_alpha_i"] = _tc[ret_col] - (_rfr_trade + beta * (_tc[bm_col] - _rfr_trade))
            alpha           = _tc["_alpha_i"].mean()
        else:
            alpha = np.nan

        port_series = ds_up["unleveraged_total_portfolio_value"]
        max_dd      = ((port_series - port_series.cummax()) / port_series.cummax()).min()

        daily_metrics_rows.append({
            "date":                         current_date,
            "horizon":                      h,
            # returns
            "simple_ret_%":                 round(100 * simple_ret, 4),
            "cash_on_cash_ret_%":           round(100 * cash_on_cash_ret, 4),
            "avg_trade_ret_%":              round(100 * avg_trade_ret, 4),
            "med_trade_ret_%":              round(100 * med_trade_ret, 4),
            "win_rate_%":                   round(100 * win_rate, 4),
            "avg_win_%":                    round(100 * avg_win, 4),
            "avg_loss_%":                   round(100 * avg_loss, 4),
            # risk
            "volatility_of_returns_%":      round(100 * vol_returns, 4),
            "sharpe_ratio":                 round(sharpe, 4),
            "max_drawdown_%":               round(100 * max_dd, 4),
            "beta":                         round(beta, 4),
            # benchmark
            "buy_hold_ret_%":               round(100 * buy_hold_ret_all, 4),
            "buy_hold_ret_long_only_%":     round(100 * buy_hold_ret_long, 4),
            "alpha_%":                      round(100 * alpha, 4),
            # counts
            "n_trades":                     len(trade_rets),
            "n_winning":                    len(winning),
            "n_losing":                     len(losing),
        })

metrics_df = pd.DataFrame(daily_metrics_rows)

# ============================================================
# STEP 9 — SAVE OUTPUTS TO DASHBOARD OUTPUTS
# ============================================================
import os
from datetime import datetime

_run_ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
_run_id  = _os.environ.get("RUN_ID", _run_ts)
_out_dir = os.path.join("Dashboard Outputs", f"run_{_run_id}")
os.makedirs(_out_dir, exist_ok=True)

df_lasso_dec.to_csv(os.path.join(_out_dir, "lasso_forecasts.csv"), index=False)
strat_df.to_csv(os.path.join(_out_dir, "strat_df.csv"), index=False)

for h in HORIZONS:
    all_daily_summaries[h].to_csv(
        os.path.join(_out_dir, f"daily_summary_h{h}.csv"), index=False
    )

summary_df.to_csv(os.path.join(_out_dir, "summary_df.csv"))
metrics_df.to_csv(os.path.join(_out_dir, "metrics_df.csv"))

print(f"Outputs saved to: {_out_dir}")

# ── Additional dashboard analytics ───────────────────────────────────────────
# Compute realized volatility per (ticker, date_, horizon) from yahoo_hist
_y = yahoo_hist.copy()
_y["date_"] = pd.to_datetime(_y["date_"], errors="coerce").dt.normalize()
_y = _y.sort_values(["ticker","date_"])
_y["ret_1"] = np.log(_y["close_"] / _y.groupby("ticker")["close_"].shift(1))
_dec_dates = pd.to_datetime(strat_df["date_"].unique())
_rv_rows = []
for _tkr, _grp in _y.groupby("ticker"):
    _gi = _grp.set_index("date_").sort_index()
    for _d in _dec_dates:
        _d = pd.Timestamp(_d)
        for _h in HORIZONS:
            _fr = _gi.loc[_gi.index > _d, "ret_1"].iloc[:_h]
            if len(_fr) == _h:
                _rv_rows.append({"ticker": _tkr, "date_": _d.strftime("%Y-%m-%d"),
                                  "horizon": _h, "realized_vol": float(_fr.std() * np.sqrt(_h))})
rv_dec = pd.DataFrame(_rv_rows)
rv_dec.to_csv(os.path.join(_out_dir, "rv_dec.csv"), index=False)

# VRP = IV - realized_vol
_vrp_rows = []
for _h in HORIZONS:
    _iv_col = f"iv_hday_{_h}"
    if _iv_col not in strat_df.columns:
        continue
    _sub = strat_df[["ticker","date_",_iv_col]].dropna(subset=[_iv_col]).copy()
    _sub = _sub.rename(columns={_iv_col: "iv_hday"})
    _sub["horizon"] = _h
    _sub["date_"] = _sub["date_"].astype(str)
    _rv_h = rv_dec[rv_dec["horizon"]==_h][["ticker","date_","realized_vol"]].copy()
    _rv_h["date_"] = _rv_h["date_"].astype(str)
    _sub = _sub.merge(_rv_h, on=["ticker","date_"], how="left")
    _sub["vrp"] = _sub["iv_hday"] - _sub["realized_vol"]
    _vrp_rows.append(_sub)
if _vrp_rows:
    pd.concat(_vrp_rows, ignore_index=True).to_csv(os.path.join(_out_dir, "vrp_df.csv"), index=False)

# Calibration data: Lasso predicted vs realized + IV
_cal_rows = []
for _h in HORIZONS:
    _fig_col = f"rfwd_tplus{_h}"
    _iv_col  = f"iv_hday_{_h}"
    if _fig_col not in strat_df.columns:
        continue
    _sub = strat_df[["ticker","date_",_fig_col,_iv_col]].dropna(subset=[_fig_col]).copy()
    _sub = _sub.rename(columns={_fig_col: "lasso_pred", _iv_col: "iv_hday"})
    _sub["horizon"] = _h
    _sub["date_"] = _sub["date_"].astype(str)
    _rv_h = rv_dec[rv_dec["horizon"]==_h][["ticker","date_","realized_vol"]].copy()
    _rv_h["date_"] = _rv_h["date_"].astype(str)
    _sub = _sub.merge(_rv_h, on=["ticker","date_"], how="left")
    _sub["err_lasso"] = _sub["lasso_pred"] - _sub["realized_vol"]
    _sub["err_iv"]    = _sub["iv_hday"]    - _sub["realized_vol"]
    _sub["ae_lasso"]  = _sub["err_lasso"].abs()
    _sub["ae_iv"]     = _sub["err_iv"].abs()
    _cal_rows.append(_sub)
if _cal_rows:
    pd.concat(_cal_rows, ignore_index=True).to_csv(os.path.join(_out_dir, "cal_df.csv"), index=False)

# Transaction cost data
_tc_rows = []
for _h in HORIZONS:
    _ask_e = f"straddle_ask_entry_{_h}"; _bid_e = f"straddle_bid_entry_{_h}"
    _ask_x = f"straddle_ask_exit_{_h}";  _bid_x = f"straddle_bid_exit_{_h}"
    _sig_col = f"signal_{_h}"; _ret_col = f"trade_ret_{_h}"
    if not all(c in strat_df.columns for c in [_ask_e,_bid_e,_ask_x,_bid_x,_sig_col,_ret_col]):
        continue
    _traded = strat_df[strat_df[_sig_col]!=0].dropna(subset=[_ask_e,_bid_e,_ask_x,_bid_x,_ret_col]).copy()
    if len(_traded) == 0:
        continue
    _traded["horizon"] = _h
    _traded["mid_entry"] = (_traded[_ask_e] + _traded[_bid_e]) / 2
    _traded["mid_exit"]  = (_traded[_ask_x] + _traded[_bid_x]) / 2
    _me = _traded["mid_entry"].replace(0, np.nan)
    _mx = _traded["mid_exit"].replace(0, np.nan)
    _traded["mid_ret"]          = np.where(_traded[_sig_col]==1,
        (_traded["mid_exit"]-_traded["mid_entry"])/_me,
        (_traded["mid_entry"]-_traded["mid_exit"])/_me)
    _traded["actual_ret"]        = _traded[_ret_col]
    _traded["tc_drag"]           = _traded["actual_ret"] - _traded["mid_ret"]
    _traded["entry_spread_pct"]  = (_traded[_ask_e]-_traded[_bid_e])/_me*100
    _traded["exit_spread_pct"]   = (_traded[_ask_x]-_traded[_bid_x])/_mx*100
    _traded["roundtrip_pct"]     = _traded["entry_spread_pct"] + _traded["exit_spread_pct"]
    _tc_rows.append(_traded[["ticker","date_","horizon","mid_ret","actual_ret",
                               "tc_drag","entry_spread_pct","exit_spread_pct",
                               "roundtrip_pct",_sig_col]].rename(columns={_sig_col:"signal"}))
if _tc_rows:
    pd.concat(_tc_rows, ignore_index=True).to_csv(os.path.join(_out_dir, "tc_df.csv"), index=False)

# Extended strat_df with Lasso forecasts and IV columns for charts
_ext_keep = ["ticker","date_"]
if "spot" in strat_df.columns:
    _ext_keep.append("spot")
for _h in HORIZONS:
    for _c in [f"signal_{_h}", f"trade_ret_{_h}", f"pnl_{_h}",
               f"exit_date_{_h}", f"vol_spread_{_h}",
               f"rfwd_tplus{_h}", f"iv_hday_{_h}"]:
        if _c in strat_df.columns:
            _ext_keep.append(_c)
strat_df[[c for c in _ext_keep if c in strat_df.columns]].to_csv(
    os.path.join(_out_dir, "strat_df_ext.csv"), index=False)

print("Dashboard analytics saved.")
