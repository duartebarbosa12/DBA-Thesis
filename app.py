from flask import Flask, render_template, request, jsonify, abort
import subprocess, threading, uuid, os, sys
import pandas as pd

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
PY   = os.path.join(BASE, ".venv", "bin", "python")

runs = {}   # run_id -> {"status": "running"|"done"|"error", "error": str|None}


def _run_algo(run_id, extras):
    env = os.environ.copy()
    env.update(extras)
    env["RUN_ID"] = run_id
    try:
        r = subprocess.run(
            [PY, "algorithm.py"],
            env=env,
            capture_output=True,
            text=True,
            cwd=BASE,
            timeout=2400,   # 40-minute hard limit
        )
        if r.returncode != 0:
            runs[run_id] = {"status": "error", "error": r.stderr[-4000:]}
        else:
            runs[run_id] = {"status": "done", "error": None}
    except subprocess.TimeoutExpired:
        runs[run_id] = {"status": "error", "error": "Algorithm timed out (>40 min)."}
    except Exception as e:
        runs[run_id] = {"status": "error", "error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    d = request.json or {}
    run_id = uuid.uuid4().hex[:10]
    runs[run_id] = {"status": "running", "error": None}
    extras = {
        "THRESHOLD_LONG":    str(d.get("threshold_long",    -0.01)),
        "THRESHOLD_SHORT":   str(d.get("threshold_short",    0.0)),
        "CAPITAL_PER_TRADE": str(d.get("capital_per_trade",  200)),
        "INITIAL_CAPITAL":   str(d.get("initial_capital",   10000)),
        "PLOWBACK_RATIO":    str(d.get("plowback_ratio",      0.0)),
        "MIN_DTE_DELTA":     str(d.get("min_dte_delta",         0)),
    }
    threading.Thread(target=_run_algo, args=(run_id, extras), daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/status/<run_id>")
def status(run_id):
    return jsonify(runs.get(run_id, {"status": "unknown", "error": None}))


@app.route("/dashboard/<run_id>")
def dashboard(run_id):
    info = runs.get(run_id, {})
    if info.get("status") != "done":
        abort(404)

    out = os.path.join(BASE, "Dashboard Outputs", f"run_{run_id}")

    def read(name, **kw):
        p = os.path.join(out, name)
        return pd.read_csv(p, **kw) if os.path.exists(p) else pd.DataFrame()

    summary  = read("summary_df.csv",  index_col=0)
    metrics  = read("metrics_df.csv",  index_col=0)
    ds_h3    = read("daily_summary_h3.csv")
    ds_h5    = read("daily_summary_h5.csv")
    ds_h10   = read("daily_summary_h10.csv")
    strat    = read("strat_df_ext.csv")
    vrp_df   = read("vrp_df.csv")
    cal_df   = read("cal_df.csv")
    tc_df    = read("tc_df.csv")
    rv_dec   = read("rv_dec.csv")

    # Trim strat to essential columns (already trimmed in strat_df_ext)
    def to_json(df):
        return df.to_json(orient="records", date_format="iso")

    return render_template(
        "dashboard.html",
        run_id=run_id,
        summary_json=to_json(summary),
        metrics_json=to_json(metrics),
        ds_h3_json=to_json(ds_h3),
        ds_h5_json=to_json(ds_h5),
        ds_h10_json=to_json(ds_h10),
        strat_json=to_json(strat),
        vrp_json=to_json(vrp_df),
        cal_json=to_json(cal_df),
        tc_json=to_json(tc_df),
        rv_json=to_json(rv_dec),
    )


if __name__ == "__main__":
    print("Starting Tradelity portal at http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
