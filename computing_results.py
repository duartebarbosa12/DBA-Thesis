##Section 0: HELPERS

# ════════════════════════════════════════════════════════════
# CELL 0 — HELPERS  (run this cell first)
# ════════════════════════════════════════════════════════════
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from google.colab import files
import warnings, os
warnings.filterwarnings("ignore")
 
# ── palette ────────────────────────────────────────────────
H_COLORS  = {3: "#4878CF", 5: "#6ACC65", 10: "#D65F5F"}
H_MARKERS = {3: "o",       5: "s",       10: "^"}
TICKER_COLORS = dict(zip(sorted(tickers),
                         plt.cm.tab10.colors[:len(tickers)]))
 
# ── render DataFrame as matplotlib figure (for PDF) ────────
def df_to_fig(df, title="", fontsize=7, col_width=1.5):
    df_s = df.copy()
    for c in df_s.select_dtypes(include=[float, np.floating]).columns:
        df_s[c] = df_s[c].round(4)
    df_s = df_s.astype(str).replace("nan","—").replace("<NA>","—")
    nr, nc = df_s.shape
    fw = max(8, nc * col_width)
    fh = max(2, nr * 0.30 + 1.2)
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.axis("off")
    tbl = ax.table(cellText=df_s.values,
                   colLabels=df_s.columns.tolist(),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.auto_set_column_width(list(range(nc)))
    for j in range(nc):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, nr+1):
        bg = "#f4f4f4" if i % 2 == 0 else "white"
        for j in range(nc): tbl[i, j].set_facecolor(bg)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig
 
# ── save PDF ───────────────────────────────────────────────
def save_pdf(figs, fname):
    with PdfPages(fname) as pdf:
        for fig in figs:
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
    files.download(fname)
    print(f"✓ {fname}")
 
# ── save CSVs ──────────────────────────────────────────────
def save_csvs(dct, prefix):
    for name, df in dct.items():
        fname = f"{prefix}_{name}.csv"
        df.to_csv(fname, index=False)
        files.download(fname)
        print(f"✓ {fname}")
 
# ── build realized vol for all (ticker, date, horizon) ────
def build_rv_dec(horizons=HORIZONS):
    y = yahoo_hist.copy()
    y["date_"] = pd.to_datetime(y["date_"], errors="coerce").dt.normalize()
    y = y.sort_values(["ticker","date_"])
    y["ret_1"] = np.log(y["close_"] / y.groupby("ticker")["close_"].shift(1))
    dec_dates = pd.to_datetime(strat_df["date_"].unique())
    rows = []
    for tkr, grp in y.groupby("ticker"):
        if tkr not in tickers: continue
        gi = grp.set_index("date_").sort_index()
        for d in dec_dates:
            d = pd.Timestamp(d)
            for h in horizons:
                fr = gi.loc[gi.index > d, "ret_1"].iloc[:h]
                if len(fr) == h:
                    rows.append({"ticker": tkr, "date_": d,
                                 "horizon": h,
                                 "realized_vol": fr.std() * np.sqrt(h)})
    return pd.DataFrame(rows)
 
rv_dec = build_rv_dec()
print(f"✓ rv_dec built: {len(rv_dec):,} rows")


##Section 2: VOLATILITY RISK PREMIUM IN DECEMBER 2023

_f2, _c2 = [], {}
 
# Build VRP = IV_hday - realized_vol
vrp_rows = []
for h in HORIZONS:
    iv_col = f"iv_hday_{h}"
    sub = strat_df[["ticker","date_",iv_col]].dropna(subset=[iv_col]).copy()
    sub = sub.rename(columns={iv_col: "iv_hday"})
    sub["horizon"] = h
    rv_h = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    sub = sub.merge(rv_h, on=["ticker","date_"], how="left")
    sub["vrp"] = sub["iv_hday"] - sub["realized_vol"]
    vrp_rows.append(sub)
vrp_df = pd.concat(vrp_rows, ignore_index=True)
vrp_df["date_"] = pd.to_datetime(vrp_df["date_"])
 
# 2a — Heatmap: ticker × date, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
for ax, h in zip(axes, HORIZONS):
    sub = vrp_df[vrp_df["horizon"]==h].copy()
    sub["ds"] = sub["date_"].dt.strftime("%b %d")
    piv = sub.pivot_table(index="ticker", columns="ds",
                           values="vrp", aggfunc="mean")
    col_order = (sub[["date_","ds"]].drop_duplicates()
                 .sort_values("date_")["ds"].tolist())
    piv = piv[[c for c in col_order if c in piv.columns]]
    vmax = vrp_df["vrp"].abs().quantile(0.95)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="VRP (IV − Realized)")
fig.suptitle("Volatility Risk Premium: IV − Realized Vol  |  December 2023",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f2.append(fig); plt.show()
 
# 2b — Average VRP per ticker, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    sub = (vrp_df[vrp_df["horizon"]==h]
           .groupby("ticker", as_index=False)["vrp"].mean()
           .sort_values("vrp", ascending=False))
    bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["vrp"]]
    ax.bar(sub["ticker"], sub["vrp"], color=bar_c, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Avg VRP (IV − Realized Vol)")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Average Volatility Risk Premium per Ticker  |  December 2023",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f2.append(fig); plt.show()
 
# 2c — VRP time series through December, one line per horizon
vrp_ts = (vrp_df.groupby(["date_","horizon"])["vrp"].mean().reset_index())
fig, ax = plt.subplots(figsize=(14, 5))
for h in HORIZONS:
    sub = vrp_ts[vrp_ts["horizon"]==h].sort_values("date_")
    ax.plot(sub["date_"], sub["vrp"], color=H_COLORS[h],
            marker=H_MARKERS[h], label=f"h={h}", linewidth=2, markersize=5)
ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Average VRP across tickers (IV − Realized)")
ax.set_title("Volatility Risk Premium Time Series  |  December 2023",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
_f2.append(fig); plt.show()
 
# 2d — VRP summary table
vrp_summary = (vrp_df.groupby(["ticker","horizon"])["vrp"]
               .agg(mean="mean", median="median", std="std",
                    min="min", max="max",
                    pct_positive=lambda x: round((x > 0).mean()*100, 1))
               .reset_index().round(5))
fig = df_to_fig(vrp_summary, "VRP Summary Statistics — Ticker × Horizon")
_f2.append(fig); plt.show()
_c2["vrp_summary"] = vrp_summary
_c2["vrp_full"]    = vrp_df[["ticker","date_","horizon","iv_hday",
                               "realized_vol","vrp"]]
 
save_pdf(_f2, "section2_volatility_risk_premium.pdf")
save_csvs(_c2, "section2")

## Section 3: FIGARCH FORECAST CALIBRATION


_f3, _c3 = [], {}
 
# Build calibration dataframe
cal_rows = []
for h in HORIZONS:
    fig_col = f"rfwd_tplus{h}"
    iv_col  = f"iv_hday_{h}"
    sub = strat_df[["ticker","date_",fig_col,iv_col]].dropna(subset=[fig_col]).copy()
    sub = sub.rename(columns={fig_col:"figarch_pred", iv_col:"iv_hday"})
    sub["horizon"] = h
    rv_h = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    sub  = sub.merge(rv_h, on=["ticker","date_"], how="left")
    sub["err_figarch"] = sub["figarch_pred"] - sub["realized_vol"]
    sub["err_iv"]      = sub["iv_hday"]      - sub["realized_vol"]
    sub["ae_figarch"]  = sub["err_figarch"].abs()
    sub["ae_iv"]       = sub["err_iv"].abs()
    cal_rows.append(sub)
cal_df = pd.concat(cal_rows, ignore_index=True)
 
# 3a — Calibration plot: decile bins of predicted vs realized
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, h in zip(axes, HORIZONS):
    sub = cal_df[cal_df["horizon"]==h].dropna(subset=["figarch_pred","realized_vol"])
    sub = sub.copy()
    sub["decile"] = pd.qcut(sub["figarch_pred"], q=10,
                             labels=False, duplicates="drop")
    binned = sub.groupby("decile")[["figarch_pred","realized_vol"]].mean().reset_index()
    ax.scatter(binned["figarch_pred"], binned["realized_vol"],
               color=H_COLORS[h], s=90, zorder=5, label="Decile avg")
    # 45° line
    lo = min(binned["figarch_pred"].min(), binned["realized_vol"].min()) * 0.95
    hi = max(binned["figarch_pred"].max(), binned["realized_vol"].max()) * 1.05
    ax.plot([lo,hi],[lo,hi],"k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("FIGARCH Predicted Vol (decile avg)", fontsize=10)
    ax.set_ylabel("Realized Vol (decile avg)", fontsize=10)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle("FIGARCH Calibration — Predicted vs Realized (by Decile of Predicted)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f3.append(fig); plt.show()
 
# 3b — Forecast error vs prior-week vol
y_tmp = yahoo_hist.copy()
y_tmp["date_"] = pd.to_datetime(y_tmp["date_"], errors="coerce").dt.normalize()
y_tmp = y_tmp.sort_values(["ticker","date_"])
y_tmp["ret_1"] = np.log(y_tmp["close_"] /
                         y_tmp.groupby("ticker")["close_"].shift(1))
y_tmp["rv5_prior"] = (y_tmp.groupby("ticker")["ret_1"]
                      .transform(lambda s: s.shift(1).rolling(5)
                                  .apply(lambda x: np.sqrt((x**2).sum()))))
prior_vol = y_tmp[["ticker","date_","rv5_prior"]].copy()
 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    sub = (cal_df[cal_df["horizon"]==h]
           .merge(prior_vol, on=["ticker","date_"], how="left")
           .dropna(subset=["err_figarch","rv5_prior"]))
    ax.scatter(sub["rv5_prior"], sub["err_figarch"],
               alpha=0.35, color=H_COLORS[h], s=15)
    z  = np.polyfit(sub["rv5_prior"], sub["err_figarch"], 1)
    xs = np.linspace(sub["rv5_prior"].min(), sub["rv5_prior"].max(), 100)
    ax.plot(xs, np.polyval(z, xs), "k-", linewidth=2,
            label=f"Trend (slope={z[0]:.2f})")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Prior 5-day Realized Vol", fontsize=10)
    ax.set_ylabel("FIGARCH Forecast Error (Predicted − Realized)", fontsize=10)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle("FIGARCH Forecast Error vs Prior Market Volatility",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f3.append(fig); plt.show()
 
# 3c — FIGARCH MAE vs IV MAE per ticker, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, h in zip(axes, HORIZONS):
    sub = (cal_df[cal_df["horizon"]==h]
           .groupby("ticker")[["ae_figarch","ae_iv"]].mean()
           .reset_index().sort_values("ae_figarch"))
    x = np.arange(len(sub)); w = 0.35
    ax.bar(x - w/2, sub["ae_figarch"], w, label="FIGARCH MAE",
           color=H_COLORS[h], edgecolor="white")
    ax.bar(x + w/2, sub["ae_iv"],      w, label="IV MAE",
           color="#aaaaaa", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["ticker"], rotation=45, ha="right")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle("FIGARCH vs Implied Vol: Forecast MAE per Ticker",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f3.append(fig); plt.show()
 
# 3d — Calibration accuracy table
acc_tbl = (cal_df.groupby(["ticker","horizon"])
           .agg(figarch_mae=("ae_figarch","mean"),
                figarch_bias=("err_figarch","mean"),
                iv_mae=("ae_iv","mean"),
                iv_bias=("err_iv","mean"),
                n=("ae_figarch","count"))
           .reset_index().round(5))
fig = df_to_fig(acc_tbl, "FIGARCH vs IV — Accuracy & Bias per Ticker × Horizon")
_f3.append(fig); plt.show()
_c3["calibration_accuracy"] = acc_tbl
_c3["calibration_full"] = cal_df[["ticker","date_","horizon",
                                   "figarch_pred","iv_hday","realized_vol",
                                   "err_figarch","err_iv","ae_figarch","ae_iv"]]
 
save_pdf(_f3, "section3_figarch_calibration.pdf")
save_csvs(_c3, "section3")

## Section 4: SIGNAL GENERATION & TRADE ACTIVITY

_f4, _c4 = [], {}
 
# 4a — Signal counts & pipeline diagnostics → tables
sig_rows, diag_rows = [], []
for h in HORIZONS:
    sig     = strat_df[f"signal_{h}"]
    ask_e   = f"straddle_ask_entry_{h}"
    bid_x   = f"straddle_bid_exit_{h}"
    ret_col = f"trade_ret_{h}"
    n_long    = int((sig == 1).sum())
    n_short   = int((sig ==-1).sum())
    n_neutral = int((sig == 0).sum())
    avg_ret_l = strat_df.loc[(sig==1)  & strat_df[ret_col].notna(), ret_col].mean()
    avg_ret_s = strat_df.loc[(sig==-1) & strat_df[ret_col].notna(), ret_col].mean()
    sig_rows.append({
        "horizon":h, "n_long":n_long, "n_short":n_short, "n_neutral":n_neutral,
        "total_signal≠0": n_long+n_short,
        "ask_entry_ok":   int(strat_df[ask_e].notna().sum()),
        "bid_exit_ok":    int(strat_df[bid_x].notna().sum()),
        "both_quotes_ok": int(strat_df[[ask_e,bid_x]].notna().all(axis=1).sum()),
        "avg_ret_long":   round(float(avg_ret_l),4) if not np.isnan(avg_ret_l) else np.nan,
        "avg_ret_short":  round(float(avg_ret_s),4) if not np.isnan(avg_ret_s) else np.nan,
    })
    diag_rows.append({
        "horizon":h,
        "total_rows":      int(len(strat_df)),
        "has_spot":        int(strat_df["spot"].notna().sum()),
        "has_exit_date":   int(strat_df[f"exit_date_{h}"].notna().sum())      if f"exit_date_{h}"   in strat_df.columns else "—",
        "has_contract":    int(strat_df[f"optid_call_{h}"].notna().sum())     if f"optid_call_{h}"  in strat_df.columns else "—",
        "has_iv":          int(strat_df[f"iv_ann_{h}"].notna().sum())         if f"iv_ann_{h}"      in strat_df.columns else "—",
        "has_entry_quote": int(strat_df[ask_e].notna().sum()),
        "has_vol_spread":  int(strat_df[f"vol_spread_{h}"].notna().sum()),
    })
 
sig_counts_df = pd.DataFrame(sig_rows)
diag_df       = pd.DataFrame(diag_rows)
fig = df_to_fig(sig_counts_df, "Signal Counts, Quote Availability & Average Return by Horizon")
_f4.append(fig); plt.show()
fig = df_to_fig(diag_df, "Pipeline Diagnostics — Data Availability per Horizon")
_f4.append(fig); plt.show()
_c4["signal_counts"]       = sig_counts_df
_c4["pipeline_diagnostics"] = diag_df
 
# 4b — Vol spread descriptive stats table (from existing Cell 112 logic)
spread_stats_rows = []
for h in HORIZONS:
    s = strat_df[f"vol_spread_{h}"].dropna()
    spread_stats_rows.append({
        "horizon":h,
        "mean": round(s.mean(),5), "median": round(s.median(),5),
        "std":  round(s.std(),5),  "min": round(s.min(),5),
        "max":  round(s.max(),5),
        "|spread|>0.005": int((s.abs()>0.005).sum()),
        "|spread|>0.010": int((s.abs()>0.010).sum()),
        "|spread|>0.020": int((s.abs()>0.020).sum()),
    })
spread_stats_df = pd.DataFrame(spread_stats_rows)
fig = df_to_fig(spread_stats_df, "Vol Spread Descriptive Statistics by Horizon")
_f4.append(fig); plt.show()
_c4["spread_statistics"] = spread_stats_df
 
# 4c — Histogram of vol_spread (existing Cell 117, kept exactly)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, h in zip(axes, HORIZONS):
    spreads = strat_df[f"vol_spread_{h}"].dropna()
    ax.hist(spreads, bins=20, color=H_COLORS[h], edgecolor="white", alpha=0.85)
    ax.axvline(0,               color="black", linewidth=1.5, linestyle="-",  label="zero")
    ax.axvline(THRESHOLD_LONG,  color="red",   linewidth=1.5, linestyle="--",
               label=f"long ({THRESHOLD_LONG})")
    ax.axvline(-THRESHOLD_SHORT,color="blue",  linewidth=1.5, linestyle="--",
               label=f"short ({-THRESHOLD_SHORT})")
    n_l = (spreads >  THRESHOLD_LONG).sum()
    n_s = (spreads < -THRESHOLD_SHORT).sum()
    n_n = len(spreads) - n_l - n_s
    ax.set_title(f"h={h}  |  long={n_l}  neutral={n_n}  short={n_s}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Vol Spread (FIGARCH − IV)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle("Distribution of Volatility Spreads by Horizon",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_f4.append(fig); plt.show()
 
# 4d — Signal calendar heatmap: date × ticker, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
for ax, h in zip(axes, HORIZONS):
    tmp = strat_df[["ticker","date_",f"signal_{h}"]].copy()
    tmp["ds"] = pd.to_datetime(tmp["date_"]).dt.strftime("%b %d")
    piv = tmp.pivot_table(index="ticker", columns="ds",
                           values=f"signal_{h}", aggfunc="first")
    col_order = (tmp[["date_","ds"]].drop_duplicates()
                 .assign(date_=pd.to_datetime(tmp["date_"]))
                 .sort_values("date_")["ds"].unique().tolist())
    piv = piv[[c for c in col_order if c in piv.columns]]
    im  = ax.imshow(piv.values.astype(float), aspect="auto",
                    cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=axes[-1], label="-1=Short  0=Neutral  +1=Long", ticks=[-1,0,1])
fig.suptitle("Signal Calendar — December 2023  (Green=Long, Yellow=Neutral, Red=Short)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_f4.append(fig); plt.show()
 
# 4e — Stacked bar per ticker: long / short / neutral
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    sig_col = f"signal_{h}"
    counts  = (strat_df.groupby("ticker")[sig_col]
               .value_counts().unstack(fill_value=0))
    for c in [1,-1,0]:
        if c not in counts.columns: counts[c] = 0
    counts = counts[[1,-1,0]]
    ax.bar(counts.index, counts[1],  color="#2ecc71", label="Long",    edgecolor="white")
    ax.bar(counts.index, counts[-1], color="#e74c3c", label="Short",   edgecolor="white",
           bottom=counts[1].values)
    ax.bar(counts.index, counts[0],  color="#bdc3c7", label="Neutral", edgecolor="white",
           bottom=(counts[1]+counts[-1]).values)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Number of Days")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Signal Distribution per Ticker  |  December 2023",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f4.append(fig); plt.show()
 
save_pdf(_f4, "section4_signal_generation.pdf")
save_csvs(_c4, "section4")

## Section 5: TRANSACTION COST DECOMPOSITION

_f5, _c5 = [], {}
 
# Build TC dataframe
tc_rows = []
for h in HORIZONS:
    ask_e = f"straddle_ask_entry_{h}"; bid_e = f"straddle_bid_entry_{h}"
    ask_x = f"straddle_ask_exit_{h}";  bid_x = f"straddle_bid_exit_{h}"
    sig_col = f"signal_{h}";            ret_col = f"trade_ret_{h}"
    liq_col = f"liquidity_filtered_{h}"
 
    traded = (strat_df[strat_df[sig_col] != 0]
              .dropna(subset=[ask_e, bid_e, ask_x, bid_x, ret_col])
              .copy())
    traded["horizon"]    = h
    traded["mid_entry"]  = (traded[ask_e] + traded[bid_e]) / 2
    traded["mid_exit"]   = (traded[ask_x] + traded[bid_x]) / 2
    traded["mid_ret"]    = np.where(
        traded[sig_col] == 1,
        (traded["mid_exit"] - traded["mid_entry"]) / traded["mid_entry"],
        (traded["mid_entry"] - traded["mid_exit"]) / traded["mid_entry"])
    traded["actual_ret"]        = traded[ret_col]
    traded["tc_drag"]           = traded["actual_ret"] - traded["mid_ret"]
    traded["entry_spread_pct"]  = (traded[ask_e] - traded[bid_e]) / traded["mid_entry"] * 100
    traded["exit_spread_pct"]   = (traded[ask_x] - traded[bid_x]) / traded["mid_exit"]  * 100
    traded["roundtrip_pct"]     = traded["entry_spread_pct"] + traded["exit_spread_pct"]
    tc_rows.append(traded[["ticker","date_","horizon","mid_ret","actual_ret",
                             "tc_drag","entry_spread_pct","exit_spread_pct",
                             "roundtrip_pct",sig_col]]
                   .rename(columns={sig_col:"signal"}))
tc_df = pd.concat(tc_rows, ignore_index=True)
 
# 5a — Bar: mid-to-mid vs actual return vs TC drag per horizon
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(HORIZONS)); w = 0.26
ax.bar(x-w, [tc_df[tc_df["horizon"]==h]["mid_ret"].mean()    for h in HORIZONS],
       w, label="Mid-to-mid return", color="#4878CF", edgecolor="white")
ax.bar(x,   [tc_df[tc_df["horizon"]==h]["actual_ret"].mean() for h in HORIZONS],
       w, label="Actual return",     color="#6ACC65", edgecolor="white")
ax.bar(x+w, [tc_df[tc_df["horizon"]==h]["tc_drag"].mean()    for h in HORIZONS],
       w, label="TC drag (actual − mid)", color="#e74c3c", edgecolor="white")
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=11)
ax.set_ylabel("Average Return"); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
ax.set_title("Mid-to-Mid vs Actual Return — Transaction Cost Impact",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_f5.append(fig); plt.show()
 
# 5b — Scatter: round-trip cost % vs trade return, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    sub = tc_df[tc_df["horizon"]==h].dropna(subset=["roundtrip_pct","actual_ret"])
    for sig_val, label, color in [(1,"Long","#2ecc71"),(-1,"Short","#e74c3c")]:
        d = sub[sub["signal"]==sig_val]
        if len(d): ax.scatter(d["roundtrip_pct"], d["actual_ret"],
                              color=color, alpha=0.7, s=40, label=label,
                              edgecolor="black", linewidth=0.3)
    if len(sub) >= 3:
        valid = sub[["roundtrip_pct","actual_ret"]].dropna()
        z  = np.polyfit(valid["roundtrip_pct"], valid["actual_ret"], 1)
        xs = np.linspace(valid["roundtrip_pct"].min(), valid["roundtrip_pct"].max(), 100)
        ax.plot(xs, np.polyval(z, xs), "k--", linewidth=1.8,
                label=f"Trend (slope={z[0]:.3f})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Round-trip Spread Cost (% of entry mid)")
    ax.set_ylabel("Actual Trade Return")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle("Round-trip Transaction Cost vs Trade Return",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f5.append(fig); plt.show()
 
# 5c — Box: entry vs exit spread % per horizon
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, h in zip(axes, HORIZONS):
    sub = tc_df[tc_df["horizon"]==h]
    data   = [sub["entry_spread_pct"].dropna().values,
              sub["exit_spread_pct"].dropna().values]
    bp = ax.boxplot(data, labels=["Entry spread %","Exit spread %"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(H_COLORS[h]);  bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#999999");     bp["boxes"][1].set_alpha(0.75)
    ax.set_ylabel("Bid-Ask Spread (% of mid-price)")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Entry vs Exit Bid-Ask Spread as % of Mid-Price",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f5.append(fig); plt.show()
 
# 5d — TC summary table (replaces/extends text from Cells 112-114)
tc_summary = (tc_df.groupby("horizon")
              .agg(n_trades=("actual_ret","count"),
                   avg_mid_ret=("mid_ret","mean"),
                   avg_actual_ret=("actual_ret","mean"),
                   avg_tc_drag=("tc_drag","mean"),
                   avg_entry_spread_pct=("entry_spread_pct","mean"),
                   avg_exit_spread_pct=("exit_spread_pct","mean"),
                   avg_roundtrip_pct=("roundtrip_pct","mean"),
                   median_roundtrip_pct=("roundtrip_pct","median"),
                   max_roundtrip_pct=("roundtrip_pct","max"))
              .reset_index().round(4))
fig = df_to_fig(tc_summary, "Transaction Cost Summary per Horizon")
_f5.append(fig); plt.show()
_c5["tc_summary"] = tc_summary
_c5["tc_full"]    = tc_df
 
save_pdf(_f5, "section5_transaction_costs.pdf")
save_csvs(_c5, "section5")

## Section 6: TRADE SAMPLE INSPECTION

_f6, _c6 = [], {}
 
# 6a — Strategy performance summary
_sdf = summary_df.reset_index() if summary_df.index.name else summary_df.copy()
fig  = df_to_fig(_sdf, "Strategy Performance Summary", col_width=2.0)
_f6.append(fig); plt.show()
_c6["summary_df"] = _sdf
 
# 6b — Performance metrics
_mdf = metrics_df.reset_index() if metrics_df.index.name else metrics_df.copy()
fig  = df_to_fig(_mdf, "Step 8 — Performance Metrics", col_width=2.0)
_f6.append(fig); plt.show()
_c6["metrics_df"] = _mdf
 
# 6c — Daily portfolio summary for each horizon
for h in HORIZONS:
    daily = all_daily_summaries[h].copy()
    # select the most readable columns
    cols_show = ["date","beg_total_portfolio_value",
                 "n_long_opened","n_short_opened",
                 "n_long_closed","n_short_closed",
                 "long_gains_obtained","short_gains_obtained",
                 "long_losses","short_losses",
                 "end_total_portfolio_value",
                 "unleveraged_total_portfolio_value"]
    cols_show = [c for c in cols_show if c in daily.columns]
    fig = df_to_fig(daily[cols_show].head(30), f"Daily Portfolio Summary — h={h}")
    _f6.append(fig); plt.show()
    _c6[f"daily_summary_h{h}"] = daily
 
# 6d — Long straddle sample h=10
long10 = (strat_df[strat_df["signal_10"]==1][[
    "ticker","date_","spot","rfwd_tplus10","iv_hday_10","vol_spread_10",
    "straddle_ask_entry_10","straddle_bid_entry_10",
    "straddle_bid_exit_10","straddle_ask_exit_10",
    "exit_rolled_10","pnl_10","trade_ret_10"
]].head(15).reset_index(drop=True))
fig = df_to_fig(long10, "Long Straddles Sample — h=10")
_f6.append(fig); plt.show()
_c6["long_straddles_h10"] = long10
 
# 6e — Short straddle sample h=10
short10 = (strat_df[strat_df["signal_10"]==-1][[
    "ticker","date_","spot","rfwd_tplus10","iv_hday_10","vol_spread_10",
    "straddle_ask_entry_10","straddle_bid_entry_10",
    "straddle_bid_exit_10","straddle_ask_exit_10",
    "exit_rolled_10","pnl_10","trade_ret_10"
]].head(15).reset_index(drop=True))
fig = df_to_fig(short10, "Short Straddles Sample — h=10")
_f6.append(fig); plt.show()
_c6["short_straddles_h10"] = short10
 
# 6f — Top 10 and Bottom 10 trades h=10 (existing Cells 122/123)
for label, asc in [("Top 10 Trades by Return", False),
                   ("Bottom 10 Trades by Return", True)]:
    tbl = (strat_df[(strat_df["signal_10"]!=0) & strat_df["trade_ret_10"].notna()]
           .sort_values("trade_ret_10", ascending=asc)
           .head(10)[["ticker","date_","vol_spread_10","signal_10",
                       "straddle_ask_entry_10","straddle_bid_exit_10",
                       "exit_rolled_10","trade_ret_10"]]
           .reset_index(drop=True))
    fig = df_to_fig(tbl, f"{label} — h=10")
    _f6.append(fig); plt.show()
    _c6[label.lower().replace(" ","_")+"_h10"] = tbl
 
# NOTE: show_day_transactions("2023-12-15", h=5) — kept as existing Cell 105
# NOTE: LLY case study (Cell 120) — kept as existing cell
 
save_pdf(_f6, "section6_trade_inspection.pdf")
save_csvs(_c6, "section6")

## Section 7: RETURN ATTRIBUTION: WHAT DROVE PERFORMANCE?

_f7, _c7 = [], {}
 
# 7a — Equity curve: unleveraged_total_portfolio_value through December
fig, ax = plt.subplots(figsize=(14, 6))
for h in HORIZONS:
    ds = all_daily_summaries[h].copy()
    ds["date"] = pd.to_datetime(ds["date"])
    ax.plot(ds["date"], ds["unleveraged_total_portfolio_value"],
            color=H_COLORS[h], marker=H_MARKERS[h], markersize=5,
            linewidth=2.5, label=f"h={h}")
ax.axhline(INITIAL_CAPITAL, color="black", linewidth=1,
           linestyle="--", alpha=0.7, label=f"Initial capital ({INITIAL_CAPITAL})")
ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value ($)")
ax.set_title("Equity Curve — Unleveraged Portfolio Value  |  December 2023",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
_f7.append(fig); plt.show()
 
# 7b — Box plot: trade_ret by direction per horizon (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, h in zip(axes, HORIZONS):
    ret_col = f"trade_ret_{h}"; sig_col = f"signal_{h}"
    longs  = strat_df[(strat_df[sig_col]== 1) & strat_df[ret_col].notna()][ret_col]
    shorts = strat_df[(strat_df[sig_col]==-1) & strat_df[ret_col].notna()][ret_col]
    data   = [d.values for d in [longs, shorts] if len(d) > 0]
    labels = [l for l, d in zip(["Long","Short"],[longs,shorts]) if len(d) > 0]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], ["#2ecc71","#e74c3c"][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("Trade Return")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Trade Return Distribution by Direction",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f7.append(fig); plt.show()
 
# 7c — Average trade_ret per ticker, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    ret_col = f"trade_ret_{h}"; sig_col = f"signal_{h}"
    sub = (strat_df[strat_df[sig_col]!=0]
           .groupby("ticker")[ret_col].mean()
           .reset_index().sort_values(ret_col, ascending=False))
    bar_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[ret_col]]
    ax.bar(sub["ticker"], sub[ret_col], color=bar_c, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Avg Trade Return")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Average Trade Return per Ticker  |  Traded Positions Only",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f7.append(fig); plt.show()
 
# 7d — Scatter: vol_spread vs trade_ret, one subplot per horizon
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, HORIZONS):
    ret_col = f"trade_ret_{h}"; spread_col = f"vol_spread_{h}"
    sig_col = f"signal_{h}"
    sub = strat_df[(strat_df[sig_col]!=0) & strat_df[ret_col].notna()].copy()
    for tkr in sorted(tickers):
        d = sub[sub["ticker"]==tkr]
        if len(d):
            ax.scatter(d[spread_col], d[ret_col],
                       color=TICKER_COLORS[tkr], alpha=0.7, s=40, label=tkr,
                       edgecolor="black", linewidth=0.3)
    valid = sub[[spread_col, ret_col]].dropna()
    if len(valid) >= 3:
        z  = np.polyfit(valid[spread_col], valid[ret_col], 1)
        xs = np.linspace(valid[spread_col].min(), valid[spread_col].max(), 100)
        ax.plot(xs, np.polyval(z, xs), "k--", linewidth=2,
                label=f"Trend (slope={z[0]:.2f})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Vol Spread (FIGARCH − IV)"); ax.set_ylabel("Trade Return")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
fig.suptitle("Vol Spread vs Trade Return — Did a Larger Edge Pay Off?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f7.append(fig); plt.show()
 
# 7e — Winner / Loser summary table (replaces text from Cell 119)
wl_rows = []
for h in HORIZONS:
    ret_col    = f"trade_ret_{h}"; sig_col    = f"signal_{h}"
    spread_col = f"vol_spread_{h}"; ask_e      = f"straddle_ask_entry_{h}"
    bid_x      = f"straddle_bid_exit_{h}"
    traded  = strat_df[(strat_df[sig_col]!=0) & strat_df[ret_col].notna()].copy()
    n_total = len(traded)
    for outcome, grp in [("Winner", traded[traded[ret_col]>= 0]),
                         ("Loser",  traded[traded[ret_col]<  0])]:
        exit_entry_ratio = (grp[bid_x]/grp[ask_e]).mean() if len(grp) else np.nan
        wl_rows.append({
            "horizon":h, "outcome":outcome,
            "n": len(grp),
            "pct_of_total": f"{100*len(grp)/max(n_total,1):.1f}%",
            "avg_ret":      round(grp[ret_col].mean(), 5)      if len(grp) else np.nan,
            "avg_vol_spread": round(grp[spread_col].mean(), 5) if len(grp) else np.nan,
            "avg_entry_ask": round(grp[ask_e].mean(), 2)       if len(grp) else np.nan,
            "avg_exit_bid":  round(grp[bid_x].mean(), 2)       if len(grp) else np.nan,
            "avg_exit/entry": round(exit_entry_ratio, 4)       if not np.isnan(exit_entry_ratio) else np.nan,
        })
wl_df = pd.DataFrame(wl_rows)
fig = df_to_fig(wl_df, "Winner vs Loser Summary by Horizon")
_f7.append(fig); plt.show()
_c7["winner_loser_summary"] = wl_df
 
# 7f — Bottom 10 losers table per horizon (from Cell 119)
for h in HORIZONS:
    ret_col    = f"trade_ret_{h}"; sig_col = f"signal_{h}"
    spread_col = f"vol_spread_{h}"; ask_e   = f"straddle_ask_entry_{h}"
    bid_x      = f"straddle_bid_exit_{h}"; exit_col = f"exit_date_{h}"
    traded = strat_df[(strat_df[sig_col]!=0) & strat_df[ret_col].notna()].copy()
    losers = traded[traded[ret_col] < 0].sort_values(ret_col)
    cols   = (["ticker","date_"] +
              ([exit_col] if exit_col in strat_df.columns else []) +
              ["spot", spread_col, sig_col, ask_e, bid_x, ret_col])
    bot = losers.head(10)[cols].reset_index(drop=True)
    fig = df_to_fig(bot, f"Bottom 10 Losing Trades — h={h}")
    _f7.append(fig); plt.show()
    _c7[f"bottom10_losers_h{h}"] = bot
 
save_pdf(_f7, "section7_return_attribution.pdf")
save_csvs(_c7, "section7")

## Section 9: VOLATILITY FORECAST QUALITY

_f9, _c9 = [], {}
 
# 9a — Existing plot_vol_differences (all three horizons, aggregated)
#      These display inline via the existing function; also build PDF-safe versions
def _vol_diff_fig(h, aggregate=False, traded_only=False):
    """PDF-safe version of plot_vol_differences — returns fig instead of showing."""
    import matplotlib.patches as mpatches
    y = yahoo_hist.copy()
    y["date_"] = pd.to_datetime(y["date_"], errors="coerce").dt.normalize()
    y = y.sort_values(["ticker","date_"])
    y["ret_1"] = np.log(y["close_"] / y.groupby("ticker")["close_"].shift(1))
    realized_rows = []
    for tkr, grp in y.groupby("ticker"):
        grp = grp.set_index("date_").sort_index()
        for date in df_figarch_dec["date_"].unique():
            date = pd.Timestamp(date)
            future_rets = grp.loc[grp.index > date, "ret_1"].iloc[:h]
            if len(future_rets) == h:
                realized_rows.append({"ticker":tkr, "date_":date,
                                       f"realized_vol_{h}": future_rets.std()*np.sqrt(h)})
    df_realized = pd.DataFrame(realized_rows)
    sig_col = f"signal_{h}"; ret_col = f"trade_ret_{h}"
    figarch_col = f"rfwd_tplus{h}"; iv_col = f"iv_hday_{h}"
    plot_df = df_figarch_dec[["ticker","date_",figarch_col]].copy()
    plot_df = plot_df.merge(strat_df[["ticker","date_",sig_col,ret_col,iv_col]],
                            on=["ticker","date_"], how="left")
    plot_df = plot_df.merge(df_realized, on=["ticker","date_"], how="left")
    plot_df = plot_df.dropna(subset=[f"realized_vol_{h}", figarch_col])
    plot_df["diff_figarch"] = plot_df[figarch_col]  - plot_df[f"realized_vol_{h}"]
    plot_df["diff_iv"]      = plot_df[iv_col]        - plot_df[f"realized_vol_{h}"]
    plot_df["diff_fig_iv"]  = plot_df[figarch_col]   - plot_df[iv_col]
    plot_df["traded"]       = plot_df[ret_col].notna()
    if traded_only:
        tdates = plot_df.loc[plot_df["traded"],"date_"].unique()
        plot_df = plot_df[plot_df["date_"].isin(tdates)]
    dates   = sorted(plot_df["date_"].unique())
    tkrs    = sorted(plot_df["ticker"].unique())
    colors  = plt.cm.tab10.colors
    tc      = {t: colors[i%10] for i,t in enumerate(tkrs)}
    specs   = [
        ("diff_figarch", "FIGARCH − Realized  |  positive = FIGARCH overpredicted", False),
        ("diff_iv",      "IV − Realized  |  positive = IV overpredicted",           False),
        ("diff_fig_iv",  "FIGARCH − IV  |  positive = FIGARCH > market",            True),
    ]
    if aggregate:
        agg = (plot_df.groupby("date_")[["diff_figarch","diff_iv","diff_fig_iv"]]
               .mean().reset_index())
        fig, axes = plt.subplots(3,1, figsize=(14,12), sharex=True, sharey=True)
        for ax,(col,title,show_thr) in zip(axes,specs):
            vals = agg[col]
            bar_colors = ["tomato" if v>=0 else "steelblue" for v in vals]
            ax.bar(range(len(agg)), vals, color=bar_colors, edgecolor="white", width=0.6)
            ax.axhline(0, color="black", linewidth=0.8)
            if show_thr:
                ax.axhline(THRESHOLD_LONG,  color="green", linewidth=1.2, linestyle="--")
                ax.axhline(-THRESHOLD_SHORT,color="red",   linewidth=1.2, linestyle="--")
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Vol difference")
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels([d.strftime("%b %d") for d in agg["date_"]],
                               rotation=45, ha="right")
    else:
        n_t = len(tkrs); bw = 0.07; gap = 0.3
        xpos = np.arange(len(dates))*(n_t*bw+gap)
        fig, axes = plt.subplots(3,1, figsize=(22,14), sharex=True, sharey=True)
        for ax,(diff_col,title,show_thr) in zip(axes,specs):
            for i,tkr in enumerate(tkrs):
                tdf  = plot_df[plot_df["ticker"]==tkr].set_index("date_")
                vals = [tdf.loc[d,diff_col] if d in tdf.index else np.nan for d in dates]
                trd  = [tdf.loc[d,"traded"]  if d in tdf.index else False  for d in dates]
                x    = xpos + i*bw
                bars = ax.bar(x, vals, width=bw, color=tc[tkr], label=tkr, alpha=0.85)
                for bar, is_t in zip(bars, trd):
                    if is_t:
                        bar.set_edgecolor("black"); bar.set_linewidth(2.5)
                        bar.set_linestyle("--")
            ax.axhline(0, color="black", linewidth=0.8)
            if show_thr:
                ax.axhline(THRESHOLD_LONG,  color="green", linewidth=1.5, linestyle="--")
                ax.axhline(-THRESHOLD_SHORT,color="red",   linewidth=1.5, linestyle="--")
            ax.set_title(title, fontsize=11); ax.set_ylabel("Vol difference")
            ax.set_xticks(xpos+(n_t*bw)/2)
            ax.set_xticklabels([d.strftime("%b %d") for d in dates],
                               rotation=45, ha="right")
        ticker_handles = [mpatches.Patch(color=tc[t], label=t) for t in tkrs]
        traded_handle  = mpatches.Patch(facecolor="white", edgecolor="black",
                                         linewidth=2.5, linestyle="--",
                                         label="Traded (dashed border)")
        fig.legend(handles=ticker_handles+[traded_handle],
                   loc="upper right", fontsize=9, ncol=2)
    pos_patch = mpatches.Patch(color="tomato",    label="Positive difference")
    neg_patch = mpatches.Patch(color="steelblue", label="Negative difference")
    fig.legend(handles=[pos_patch, neg_patch], loc="upper right", fontsize=9)
    fig.suptitle(f"Volatility Differences — h={h} | "
                 f"{'Aggregated' if aggregate else 'By Ticker'} | "
                 f"{'Traded dates only' if traded_only else 'All December days'}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig
 
# Build and show all variants
for h in HORIZONS:
    fig = _vol_diff_fig(h, aggregate=True,  traded_only=False)
    _f9.append(fig); plt.show()
fig = _vol_diff_fig(h=10, aggregate=False, traded_only=True)
_f9.append(fig); plt.show()
 
# 9b — Time series overlay per ticker: FIGARCH vs IV vs Realized, one figure per horizon
for h in HORIZONS:
    fig_col = f"rfwd_tplus{h}"; iv_col = f"iv_hday_{h}"
    rv_h = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    base = (strat_df[["ticker","date_",fig_col,iv_col]].copy()
            .merge(rv_h, on=["ticker","date_"], how="left"))
    base["date_"] = pd.to_datetime(base["date_"])
    tkrs = sorted(base["ticker"].unique())
    ncols = 5; nrows = int(np.ceil(len(tkrs)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5), sharex=False)
    axes = axes.flatten()
    for idx, tkr in enumerate(tkrs):
        ax = axes[idx]
        d = base[base["ticker"]==tkr].sort_values("date_")
        ax.plot(d["date_"], d[fig_col],       color="#4878CF", linewidth=1.5,
                marker="o", markersize=3, label="FIGARCH")
        ax.plot(d["date_"], d[iv_col],        color="#e74c3c", linewidth=1.5,
                linestyle="--", marker="s", markersize=3, label="IV")
        ax.plot(d["date_"], d["realized_vol"],color="#2ecc71", linewidth=1.5,
                linestyle=":", marker="^", markersize=3, label="Realized")
        ax.set_title(tkr, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7)
    for idx in range(len(tkrs), len(axes)): axes[idx].axis("off")
    fig.suptitle(f"FIGARCH Predicted vs IV vs Realized Vol — h={h}  |  December 2023",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _f9.append(fig); plt.show()
 
# 9c — Scatter: FIGARCH predicted vs realized per horizon (one subplot per horizon)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, h in zip(axes, HORIZONS):
    fig_col = f"rfwd_tplus{h}"
    rv_h    = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    sub     = (strat_df[["ticker","date_",fig_col]].merge(rv_h, on=["ticker","date_"])
               .dropna(subset=[fig_col,"realized_vol"]))
    for tkr in sorted(tickers):
        d = sub[sub["ticker"]==tkr]
        if len(d):
            ax.scatter(d[fig_col], d["realized_vol"],
                       color=TICKER_COLORS[tkr], alpha=0.6, s=25, label=tkr,
                       edgecolor="black", linewidth=0.2)
    lo = min(sub[fig_col].min(), sub["realized_vol"].min()) * 0.9
    hi = max(sub[fig_col].quantile(0.99), sub["realized_vol"].quantile(0.99)) * 1.1
    ax.plot([lo,hi],[lo,hi],"k--", linewidth=1.5, label="45° line")
    ax.set_xlabel("FIGARCH Predicted Vol"); ax.set_ylabel("Realized Vol")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
fig.suptitle("FIGARCH Predicted vs Realized Volatility — December 2023",
             fontsize=14, fontweight="bold")
plt.tight_layout()
_f9.append(fig); plt.show()
 
# 9d — Accuracy summary table
acc_rows = []
for h in HORIZONS:
    fig_col = f"rfwd_tplus{h}"; iv_col = f"iv_hday_{h}"
    rv_h    = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    sub     = (strat_df[["ticker","date_",fig_col,iv_col]].merge(rv_h, on=["ticker","date_"])
               .dropna())
    for tkr in sub["ticker"].unique():
        d = sub[sub["ticker"]==tkr]
        acc_rows.append({
            "horizon":h, "ticker":tkr,
            "n": len(d),
            "figarch_bias": (d[fig_col]-d["realized_vol"]).mean(),
            "figarch_mae":  (d[fig_col]-d["realized_vol"]).abs().mean(),
            "figarch_rmse": ((d[fig_col]-d["realized_vol"])**2).mean()**0.5,
            "iv_bias": (d[iv_col]-d["realized_vol"]).mean(),
            "iv_mae":  (d[iv_col]-d["realized_vol"]).abs().mean(),
            "iv_rmse": ((d[iv_col]-d["realized_vol"])**2).mean()**0.5,
        })
acc_df = pd.DataFrame(acc_rows).round(6)
fig = df_to_fig(acc_df, "Vol Forecast Accuracy: FIGARCH vs IV — Ticker × Horizon")
_f9.append(fig); plt.show()
_c9["vol_accuracy"] = acc_df
 
save_pdf(_f9, "section9_vol_forecast_quality.pdf")
save_csvs(_c9, "section9")

## Section 10: WINNERS VS LOSERS DEEP-DIVE

_f10, _c10 = [], {}
 
# 10a — PDF-safe version of plot_winner_loser_vol for all horizons
def _winner_loser_fig(h=10):
    """Returns fig of winner vs loser violin plot for horizon h."""
    sig_col = f"signal_{h}"; ret_col = f"trade_ret_{h}"
    figarch_col = f"rfwd_tplus{h}"; iv_col = f"iv_hday_{h}"
    rv_h = rv_dec[rv_dec["horizon"]==h][["ticker","date_","realized_vol"]]
    df   = strat_df[["ticker","date_",sig_col,ret_col,figarch_col,iv_col]].copy()
    df   = df.merge(rv_h, on=["ticker","date_"], how="left")
    df   = df.dropna(subset=[ret_col, figarch_col, iv_col, "realized_vol"])
    df["outcome"]  = np.where(df[ret_col] > 0, "winner", "loser")
    df_long  = df[df[sig_col] ==  1]
    df_short = df[df[sig_col] == -1]
    vol_cols   = [figarch_col, iv_col, "realized_vol"]
    vol_labels = ["FIGARCH Predicted","Implied Vol (IV)","Realized Vol"]
    cw, cl = "#2ecc71", "#e74c3c"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    for row, (df_side, side_label) in enumerate([(df_long,"Long"),(df_short,"Short")]):
        winners = df_side[df_side["outcome"]=="winner"]
        losers  = df_side[df_side["outcome"]=="loser"]
        for col, (vcol, vlabel) in enumerate(zip(vol_cols, vol_labels)):
            ax = axes[row, col]
            win_v  = winners[vcol].dropna()
            loss_v = losers[vcol].dropna()
            if len(win_v) == 0 and len(loss_v) == 0: continue
            data_viol = [v for v in [win_v, loss_v] if len(v) > 0]
            positions = list(range(len(data_viol)))
            parts = ax.violinplot(data_viol, positions=positions,
                                  showmedians=True, showextrema=True)
            fc_list = [cw, cl][:len(data_viol)]
            for body, fc in zip(parts["bodies"], fc_list):
                body.set_facecolor(fc); body.set_alpha(0.7)
            for pk in ["cmedians","cmins","cmaxes","cbars"]:
                parts[pk].set_color("black"); parts[pk].set_linewidth(1.2)
            for pos, vals, fc in zip(positions, data_viol, fc_list):
                ax.scatter(np.random.normal(pos, 0.04, size=len(vals)),
                           vals, alpha=0.6, color=fc, edgecolor="black",
                           linewidth=0.4, s=35, zorder=3)
            labels_viol = ["Winners","Losers"][:len(data_viol)]
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{l} (n={len(v)})" for l,v in
                                  zip(labels_viol, data_viol)])
            if len(win_v):
                ax.hlines(win_v.mean(), -0.3,0.3, colors="darkgreen",
                          linewidth=2, linestyle="--",
                          label=f"mean={win_v.mean():.4f}")
            if len(loss_v):
                ax.hlines(loss_v.mean(), 0.7,1.3, colors="darkred",
                          linewidth=2, linestyle="--",
                          label=f"mean={loss_v.mean():.4f}")
            ax.set_title(f"{side_label} — {vlabel}", fontsize=11)
            ax.set_ylabel("Vol (h-day units)"); ax.legend(fontsize=8)
    fig.suptitle(f"Winner vs Loser Vol Distribution — h={h}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
 
for h in HORIZONS:
    fig = _winner_loser_fig(h)
    _f10.append(fig); plt.show()
 
# 10b — Quadrant scatter: vol_spread vs return, colored by outcome
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, h in zip(axes, HORIZONS):
    ret_col    = f"trade_ret_{h}"; spread_col = f"vol_spread_{h}"
    sig_col    = f"signal_{h}"
    sub = strat_df[(strat_df[sig_col]!=0) & strat_df[ret_col].notna()].copy()
    def _qcolor(row):
        if row[sig_col]== 1 and row[ret_col] >= 0: return "#1a9850"  # long winner
        if row[sig_col]== 1 and row[ret_col] <  0: return "#91cf60"  # long loser
        if row[sig_col]==-1 and row[ret_col] >= 0: return "#d73027"  # short winner
        return "#fc8d59"                                               # short loser
    sub["qc"] = sub.apply(_qcolor, axis=1)
    ax.scatter(sub[spread_col], sub[ret_col],
               c=sub["qc"], alpha=0.85, s=55, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(THRESHOLD_LONG,   color="darkgreen", linewidth=1.2, linestyle="--",
               label=f"long thr ({THRESHOLD_LONG})")
    ax.axvline(-THRESHOLD_SHORT, color="darkred",   linewidth=1.2, linestyle="--",
               label=f"short thr ({-THRESHOLD_SHORT})")
    ax.set_xlabel("Vol Spread (FIGARCH − IV)"); ax.set_ylabel("Trade Return")
    ax.set_title(f"h = {h}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
patches = [
    mpatches.Patch(color="#1a9850", label="Long  Winner"),
    mpatches.Patch(color="#91cf60", label="Long  Loser"),
    mpatches.Patch(color="#d73027", label="Short Winner"),
    mpatches.Patch(color="#fc8d59", label="Short Loser"),
]
fig.suptitle("Trade Outcome Quadrant Plot: Vol Spread vs Return",
             fontsize=14, fontweight="bold")
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5,-0.02))
plt.tight_layout()
_f10.append(fig); plt.show()
 
save_pdf(_f10, "section10_winners_losers.pdf")